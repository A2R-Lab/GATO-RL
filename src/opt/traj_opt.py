import numpy as np
import casadi as ca
from .pinocchio_template import thneed


class TrajOpt:
    def __init__(self, env, conf, w_S=0):
        self.env = env
        self.conf = conf

    def TO_Solve(self, ICS_state, init_TO_states, init_TO_controls):
        qpiters = 5
        num_iters = 100
        dt = self.conf.dt
        N = init_TO_states.shape[0]
        pyt = thneed(self.conf.URDF_PATH, N=N, dt=dt, max_qp_iters=qpiters)

        for i in range(N):
            pyt.XU[i * (self.conf.nx + self.conf.na) : i * (self.conf.nx + self.conf.na) + self.conf.nx] = init_TO_states[i,:-1]
        for i in range(N-1):
            pyt.XU[i * (self.conf.nx + self.conf.na) + self.conf.nx : (i + 1) * (self.conf.nx + self.conf.na)] = init_TO_controls[i]

        eepos_g = np.zeros(3 * pyt.N)
        eepos_g[-3:] = self.conf.goal_ee
        xs = init_TO_states[0,:-1]
        pyt.setxs(xs)

        # Run SQP optimization
        for i in range(num_iters):
            pyt.sqp(xs, eepos_g)
            xs = pyt.XU[0:self.conf.nx]
    
        X = np.array([pyt.XU[i * (self.conf.nx + self.conf.na) : i * (self.conf.nx + self.conf.na) + self.conf.nx] for i in range(N)]) 
        U = np.array([pyt.XU[i * (self.conf.nx + self.conf.na) + self.conf.nx : (i + 1) * (self.conf.nx + self.conf.na)] for i in range(N-1)])
        timesteps = init_TO_states[:, -1].reshape(N, 1)
        X = np.hstack((X, timesteps))

        return X, U
    
    def solve_constrained_SQP(self):    
        """
        Main Loop for SQP with Inequality Constraints
        """

        # Initialize params
        max_iter = int(100) # Max number of iterations of SQP (NOT timestep of the problem)
        stop_tol = 1e-5
        curr_iter = 0

        # Track constraint violation, running cost, and alpha per iteration of the solver
        constr_viol_list  = np.empty((max_iter, 1))
        running_cost_list = np.empty((max_iter, 1))
        alpha_list        = np.empty((max_iter, 1))

        # Initial random guesses
        # NOTE: make sure the x_guess initial conditions should be theta=0, w=0
        x_guess = np.zeros(((x_dim + u_dim)*self.N, 1))
        lambda_guess = np.zeros(((x_dim)*self.N, 1))
        mu_guess = np.zeros(((x_dim)*self.N, 1))
        KKT = 1

        # Filter Linear Search!
        # NOTE: The f_best and c_best should be computed from your initial guess
        #       not infinity (otherwise it might blow up on the first step)
        f_best = np.inf #np.linalg.norm(running_cost(x_guess))
        c_best = np.inf #abs(get_amnt_constr_violation(x_guess))
        alpha = 1 # initial step size
        rho = 0.5 # Shrink factor for alpha (step size)
        accept_step = False # flag for accepting step or not

        grad_f = self.grad_running_cost(x_guess)
        Hess = self.hess_running_cost(x_guess)
        g = self.get_linearized_constraints(x_guess)
        grad_g = self.get_grad_linearized_constraints(x_guess)

        while (c_best > stop_tol or             # Terminate if constraint violations is zero
            np.linalg.norm(KKT) < stop_tol): # Terminate if nabla_p L is zero
            if curr_iter >= max_iter:
                break
            
            grad_f = self.grad_running_cost(x_guess)
            Hess = self.hess_running_cost(x_guess)
            g = self.get_linearized_constraints(x_guess)
            grad_g = self.get_grad_linearized_constraints(x_guess)
            H, h = self.create_ineq_constraints(x_guess)

            # NOTE: Now we're using cvxopt to solve to consider the inequality constraints!
            problem = Problem(sparse.csr_matrix(Hess), grad_f, 
                            sparse.csr_matrix(H), h, 
                            grad_g, g)
            p_sol = solve_problem(problem, solver="cvxopt")
            x_guess_dir = p_sol.x
            lambdas_guess_dir = p_sol.y # NOTE: equality constraints multipliers
            mus_guess_dir = p_sol.z     # NOTE: inequality constraints multipliers

            # reset alpha and flag for each iteration
            alpha = 1
            accept_step = False

            while not accept_step:
                if alpha < 1e-5:
                    break
                # NOTE: Your shape was wrong dummy, the x_guess + alpha*x_guess_dir turned into a matrix
                #       hence why you need to squeeze() it
                if self.running_cost(x_guess[:,0] + alpha * x_guess_dir) < f_best:
                    f_best = self.running_cost(x_guess[:,0] + alpha * x_guess_dir)
                    accept_step = True
                if self.get_amnt_constr_violation_update_2(x_guess[:,0] + alpha * x_guess_dir) < c_best:
                    c_best = abs(self.get_amnt_constr_violation_update_2(x_guess[:,0] + alpha * x_guess_dir))
                    accept_step = True
                else:
                    alpha = alpha * rho

            # Update guesses
            x_guess = x_guess + alpha*x_guess_dir[:, np.newaxis]
            lambda_guess = (1-alpha)*lambda_guess + alpha*lambdas_guess_dir
            mu_guess = (1-alpha)*mu_guess + alpha*mus_guess_dir

            # Record constraint violation, running cost, alpha per iteration
            constr_viol_list[curr_iter]  = self.get_amnt_constr_violation_update_2(x_guess) #c_best
            running_cost_list[curr_iter] = self.running_cost(x_guess) #f_best
            alpha_list[curr_iter]        = alpha

            KKT = grad_f.squeeze() + lambdas_guess_dir @ grad_g
            print("Curr iter: ", curr_iter, " Cost: ", running_cost_list[curr_iter],
                " Constraint Violation: ", constr_viol_list[curr_iter])
            #      "KKT: ", np.linalg.norm(KKT))

            # Move onto next iteration
            curr_iter += 1

        print("Total iterations: ", curr_iter)

        # Extract values of the pendulum system
        pend_thetas = x_guess[0:num_vars:(x_dim+u_dim)]
        pend_ws     = x_guess[1:num_vars:(x_dim+u_dim)]
        pend_us     = x_guess[2:num_vars:(x_dim+u_dim)]

        plot_flag = True
        if plot_flag:
            # Trim arrays for plotting
            constr_viol_list  = constr_viol_list[:curr_iter]
            running_cost_list = running_cost_list[:curr_iter]
            alpha_list        = alpha_list[:curr_iter]

            plt.figure()
            plt.plot(np.arange(curr_iter), constr_viol_list, label="Constraint Violations")
            #plt.yscale("log")
            plt.ylabel('Constrain Violation')
            plt.xlabel('k iteration')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(curr_iter), running_cost_list, label="Running Cost")
            #plt.yscale("log")
            plt.ylabel('Running Cost')
            plt.xlabel('k iteration')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(curr_iter), alpha_list, label="Alphas")
            #plt.yscale("log")
            plt.ylabel('Alphas')
            plt.xlabel('k iteration')
            plt.grid()
            plt.legend()

            # Plot theta (angle), w (angular velocity), and u (control)
            plt.figure()
            plt.plot(np.arange(self.N), pend_thetas, label="theta (angle)")
            plt.ylabel('theta (angle)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(self.N), pend_ws, label="w (angular velocity)")
            plt.ylabel('w (angular velocity)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(self.N), pend_us, label="u (control)")
            plt.ylabel('u (control signal)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

        x_init = np.array([[.0],
                        [.0]])
        pendulum.animate_robot(x_init, pend_us.T)

        return x_guess

    def solve_unconstrained_SQP(self):
        """
        Main Loop for SQP without Inequality Constraints

        Args:
            N/A
        Returns:
            np.ndarray: Optimal trajectory x_guess
        """

        # Initialize params
        max_iter = int(20) # Max number of iterations of SQP (NOT timestep of the problem)
        stop_tol = 1e-5 # stop tolerance
        curr_iter = 0

        # Track constraint violation, running cost, and alpha per iteration of the solver
        constr_viol_list  = np.empty((max_iter, 1))
        running_cost_list = np.empty((max_iter, 1))
        alpha_list        = np.empty((max_iter, 1))

        # Initial random guesses
        # NOTE: make sure the x_guess initial conditions should be theta=0, w=0
        x_guess = np.zeros(((x_dim + u_dim)*self.N, 1))
        lambda_guess = np.zeros((self.num_eq_constraints*self.N, 1))
        KKT = 1

        # Filter Linear Search!
        # NOTE: The f_best and c_best should be computed from your initial guess
        #       not infinity (otherwise it might blow up on the first step)
        f_best = np.inf #np.linalg.norm(running_cost(x_guess))
        c_best = np.inf #abs(get_amnt_constr_violation(x_guess))
        alpha = 1 # initial step size
        rho = 0.5 # Shrink factor for alpha (step size)
        accept_step = False # flag for accepting step or not


        while (c_best > stop_tol or             # Terminate if constraint violations is zero
            np.linalg.norm(KKT) > stop_tol): # Terminate if nabla_p L is zero
            
            if curr_iter >= max_iter:
                break
            
            p_sol, H, grad_f, grad_g, g = self.construct_KKT_n_solve(x_guess)
            
            x_guess_dir = p_sol[:(x_dim+u_dim)*self.N]
            lambdas_guess_dir = p_sol[(x_dim+u_dim)*self.N:]
            
            # Filter line search
            # reset alpha and flag for each iteration
            alpha = 1
            accept_step = False

            while not accept_step:
                if alpha < 1e-5:
                    break
                # NOTE: (x_guess + alpha * x_guess_dir) has shape ((u_dim+x_dim)*self.N, 1)
                if np.linalg.norm(self.running_cost(x_guess + alpha * x_guess_dir)) < f_best:
                    f_best = self.running_cost(x_guess + alpha * x_guess_dir)
                    accept_step = True
                if self.get_amnt_constr_violation(x_guess + alpha * x_guess_dir) < c_best:
                    c_best = abs(self.get_amnt_constr_violation(x_guess + alpha * x_guess_dir))
                    accept_step = True
                else:
                    alpha = alpha * rho

            # Update guesses
            x_guess = x_guess + alpha*x_guess_dir
            lambda_guess = (1-alpha)*lambda_guess + alpha*lambdas_guess_dir

            # Record constraint violation, running cost, alpha per iteration
            constr_viol_list[curr_iter]  = self.get_amnt_constr_violation(x_guess)
            running_cost_list[curr_iter] = np.linalg.norm(self.running_cost(x_guess)) #f_best
            alpha_list[curr_iter]        = alpha

            KKT = grad_f.squeeze() + lambdas_guess_dir.T @ grad_g
            print("Curr iter: ", curr_iter, " Cost: ", running_cost_list[curr_iter],
                " Constraint Violation: ", constr_viol_list[curr_iter],
                "KKT: ", np.linalg.norm(KKT))
            
            # Move onto next iteration
            curr_iter += 1

        print("Total iterations: ", curr_iter)

        # Extract values of the pendulum system
        pend_thetas = x_guess[0:num_vars:(x_dim+u_dim)]
        pend_ws     = x_guess[1:num_vars:(x_dim+u_dim)]
        pend_us     = x_guess[2:num_vars:(x_dim+u_dim)]

        plot_flag = True
        if plot_flag:
            # Trim arrays for plotting
            constr_viol_list  = constr_viol_list[:curr_iter]
            running_cost_list = running_cost_list[:curr_iter]
            alpha_list        = alpha_list[:curr_iter]

            plt.figure()
            plt.plot(np.arange(curr_iter), constr_viol_list, label="Constraint Violations")
            #plt.yscale("log")
            plt.ylabel('Constrain Violation')
            plt.xlabel('k iteration')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(curr_iter), running_cost_list, label="Running Cost")
            #plt.yscale("log")
            plt.ylabel('Running Cost')
            plt.xlabel('k iteration')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(curr_iter), alpha_list, label="Alphas")
            #plt.yscale("log")
            plt.ylabel('Alphas')
            plt.xlabel('k iteration')
            plt.grid()
            plt.legend()

            # Plot theta (angle), w (angular velocity), and u (control)
            plt.figure()
            plt.plot(np.arange(self.N), pend_thetas, label="theta (angle)")
            plt.ylabel('theta (angle)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(self.N), pend_ws, label="w (angular velocity)")
            plt.ylabel('w (angular velocity)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(self.N), pend_us, label="u (control)")
            plt.ylabel('u (control signal)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

        x_init = np.array([[.0],
                        [.0]])
        pendulum.animate_robot(x_init, pend_us.T)

        return x_guess
