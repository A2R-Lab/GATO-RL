import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from .pinocchio_template import thneed
from scipy import sparse
from qpsolvers import solve_qp, Problem, solve_problem

# Add the confs directory to the path
# NOTE: There should be a better way to handle this, but for now we will use this
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import confs.pendulum as pendulum

class TrajOpt:
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf

    def solve_iiwa_unconstrained_SQP(self, init_traj_states, init_traj_controls):
        """
        Solve trajectory optimization for IIWA robot using unconstrained Sequential Quadratic Programming (SQP).
        
        This method uses the Pinocchio-based template to perform trajectory optimization for the IIWA robot
        arm. It initializes the optimization problem with given initial states and controls, then iteratively
        refines the trajectory to reach the desired end-effector goal position.
        
        Args:
            init_TO_states (np.ndarray): Initial trajectory states with shape (N, nx+1) where N is the 
                                       number of timesteps, nx is state dimension, and +1 is for timestep
            init_TO_controls (np.ndarray): Initial trajectory controls with shape (N-1, na) where 
                                         na is the control dimension
        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): Optimized state trajectory with shape (N, nx+1)
                - U (np.ndarray): Optimized control trajectory with shape (N-1, na)
        """
        # SQP solver parameters
        qpiters = 5        # Maximum QP iterations per SQP step
        num_iters = 100    # Maximum SQP iterations
        dt = self.conf.dt  # Time step from configuration
        N = init_traj_states.shape[0]  # Number of timesteps
        
        # Initialize Pinocchio-based trajectory optimization template
        pyt = thneed(self.conf.URDF_PATH, N=N, dt=dt, max_qp_iters=qpiters)

        # Initialize optimization variables with provided initial trajectory
        # Pack states into the optimization variable vector XU
        for i in range(N):
            pyt.XU[i * (self.conf.nx + self.conf.nu) : i * (self.conf.nx + self.conf.nu) + self.conf.nx] = init_traj_states[i,:-1]
        
        # Pack controls into the optimization variable vector XU
        for i in range(N-1):
            pyt.XU[i * (self.conf.nx + self.conf.nu) + self.conf.nx : (i + 1) * (self.conf.nx + self.conf.nu)] = init_traj_controls[i]

        # Set up end-effector goal constraint
        eepos_g = np.zeros(3 * pyt.N)  # End-effector position goals for all timesteps
        eepos_g[-3:] = self.conf.goal_ee  # Set final timestep goal to desired end-effector position
        
        # Extract initial state for warm-starting
        xs = init_traj_states[0,:-1]
        pyt.setxs(xs)

        # Run SQP optimization loop
        for i in range(num_iters):
            # Perform one SQP iteration with current state and goal
            pyt.sqp(xs, eepos_g)
            # Update current state estimate from optimization result
            xs = pyt.XU[0:self.conf.nx]
    
        # Extract optimized trajectory from solution vector
        # Unpack states from XU vector
        X = np.array([pyt.XU[i * (self.conf.nx + self.conf.nu) : i * (self.conf.nx + self.conf.nu) + self.conf.nx] for i in range(N)]) 
        # Unpack controls from XU vector
        U = np.array([pyt.XU[i * (self.conf.nx + self.conf.nu) + self.conf.nx : (i + 1) * (self.conf.nx + self.conf.nu)] for i in range(N-1)])
        
        # Reconstruct timestep information and append to states
        timesteps = init_traj_states[:, -1].reshape(N, 1)
        X = np.hstack((X, timesteps))

        return X, U
    
    def solve_pend_constrained_SQP(self, init_traj_states, init_traj_controls, display_flag=False):
        """
        Solve trajectory optimization for pendulum using constrained Sequential Quadratic Programming (SQP).
        
        This method performs trajectory optimization for a pendulum system with inequality constraints.
        It uses a filter line search approach and solves quadratic programming subproblems at each
        iteration to find the optimal trajectory that minimizes the running cost while satisfying
        the system dynamics and inequality constraints.
        
        Args:
            init_traj_states (np.ndarray): Initial trajectory states with shape (N, 2) where N is the 
                                         number of timesteps and each row contains [theta, w]
            init_traj_controls (np.ndarray): Initial trajectory controls with shape (N-1, 1) where 
                                           each element is the torque input u
            display_flag (bool, optional): Flag to display convergence plots and pendulum animation. 
                                         Defaults to False.
        
        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): Optimized state trajectory in alternating format 
                                  [theta_0, w_0, theta_1, w_1, ...] with shape (N*2, 1)
                - U (np.ndarray): Optimized control trajectory with shape (N, 1)
        
        Notes:
            - Uses cvxopt solver for handling inequality constraints
            - Implements filter line search for step size selection
            - Initial conditions are enforced to be zero (theta_0=0, w_0=0)
            - Displays convergence plots and animation if display_flag=True
        """

        # Initialize params
        max_iter = int(100) # Max number of iterations of SQP (NOT timestep of the problem)
        stop_tol = 1e-5
        curr_iter = 0
        N = init_traj_states.shape[0]  # Number of timesteps
        num_vars = (N-1)*(self.conf.nx+self.conf.nu)+self.conf.nx

        # Track constraint violation, running cost, and alpha per iteration of the solver
        constr_viol_list  = np.empty((max_iter, 1))
        running_cost_list = np.empty((max_iter, 1))
        alpha_list        = np.empty((max_iter, 1))

        # Initial random guesses
        # NOTE: make sure the x_guess initial conditions should be theta=0, w=0

        # Create combined trajectory array, interleaving states and controls:
        # [theta_0, w_0, u_0, theta_1, w_1, u_1, ...]
        x_guess = np.zeros((num_vars, 1))
        for i in range(N - 1):
            start_idx = i * (self.conf.nx + self.conf.nu)
            x_guess[start_idx:start_idx + self.conf.nx, 0] = init_traj_states[i]
            x_guess[start_idx + self.conf.nx, 0] = init_traj_controls[i, 0]
        x_guess[-2:, 0] = init_traj_states[-1]

        # Ensure initial conditions are zero (theta_0 = 0, w_0 = 0) to help with convergence
        x_guess[0, 0] = 0.0  # theta_0
        x_guess[1, 0] = 0.0  # w_0

        lambda_guess = np.zeros(((self.conf.nx)*N, 1))
        mu_guess = np.zeros(((self.conf.nx)*N, 1))

        KKT = 1

        # Filter Linear Search!
        # NOTE: The f_best and c_best should be computed from your initial guess
        #       not infinity (otherwise it might blow up on the first step)
        f_best = np.inf #np.linalg.norm(running_cost(x_guess))
        c_best = np.inf #abs(get_amnt_constr_violation(x_guess))
        alpha = 1 # initial step size
        rho = 0.5 # Shrink factor for alpha (step size)
        accept_step = False # flag for accepting step or not

        grad_f = self.env.grad_running_cost(x_guess)
        Hess = self.env.hess_running_cost(x_guess)
        g = self.env.get_linearized_constraints(x_guess)
        grad_g = self.env.get_grad_linearized_constraints(x_guess)

        while (c_best > stop_tol or             # Terminate if constraint violations is zero
            np.linalg.norm(KKT) < stop_tol): # Terminate if nabla_p L is zero
            if curr_iter >= max_iter:
                break
            
            grad_f = self.env.grad_running_cost(x_guess)
            Hess = self.env.hess_running_cost(x_guess)
            g = self.env.get_linearized_constraints(x_guess)
            grad_g = self.env.get_grad_linearized_constraints(x_guess)
            H, h = self.env.create_ineq_constraints(x_guess)

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
                if self.env.running_cost(x_guess[:,0] + alpha * x_guess_dir) < f_best:
                    f_best = self.env.running_cost(x_guess[:,0] + alpha * x_guess_dir)
                    accept_step = True
                if self.env.get_amnt_constr_violation_update_2(x_guess[:,0] + alpha * x_guess_dir) < c_best:
                    c_best = abs(self.env.get_amnt_constr_violation_update_2(x_guess[:,0] + alpha * x_guess_dir))
                    accept_step = True
                else:
                    alpha = alpha * rho

            # Update guesses
            x_guess = x_guess + alpha*x_guess_dir[:, np.newaxis]
            lambda_guess = (1-alpha)*lambda_guess + alpha*lambdas_guess_dir
            mu_guess = (1-alpha)*mu_guess + alpha*mus_guess_dir

            # Record constraint violation, running cost, alpha per iteration
            constr_viol_list[curr_iter]  = self.env.get_amnt_constr_violation_update_2(x_guess) #c_best
            running_cost_list[curr_iter] = self.env.running_cost(x_guess) #f_best
            alpha_list[curr_iter]        = alpha

            KKT = grad_f.squeeze() + lambdas_guess_dir @ grad_g
            print("Curr iter: ", curr_iter, " Cost: ", running_cost_list[curr_iter],
                " Constraint Violation: ", constr_viol_list[curr_iter])
            #      "KKT: ", np.linalg.norm(KKT))

            # Move onto next iteration
            curr_iter += 1

        print("Total iterations: ", curr_iter)

        # Extract values of the pendulum system
        pend_thetas = x_guess[0::3][:N]
        pend_ws     = x_guess[1::3][:N]
        pend_us     = x_guess[2::3]   # Already N−1
        
        # Extract states in alternating format: [theta_0, w_0, theta_1, w_1, ...]
        pend_states = np.zeros((N * self.conf.nx, 1))
        for i in range(N):
            pend_states[i * self.conf.nx, 0] = pend_thetas[i, 0]     # theta_i
            pend_states[i * self.conf.nx + 1, 0] = pend_ws[i, 0]     # w_i

        if display_flag:
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
            plt.plot(np.arange(N), pend_thetas, label="theta (angle)")
            plt.ylabel('theta (angle)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(N), pend_ws, label="w (angular velocity)")
            plt.ylabel('w (angular velocity)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(N-1), pend_us, label="u (control)")
            plt.ylabel('u (control signal)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            x_init = np.array([[.0],
                               [.0]])
            pendulum.animate_robot(x_init, pend_us.T)

        X = pend_states
        U = pend_us

        return X, U

    def solve_pend_unconstrained_SQP(self, init_traj_states, init_traj_controls, display_flag=False):
        """
        Solve trajectory optimization for pendulum using unconstrained Sequential Quadratic Programming (SQP).
        
        This method performs trajectory optimization for a pendulum system without inequality constraints.
        It solves the KKT system directly at each iteration and uses filter line search to find the
        optimal trajectory that minimizes the running cost while satisfying only the equality constraints
        (system dynamics).
        
        Args:
            init_traj_states (np.ndarray): Initial trajectory states with shape (N, 2) where N is the 
                                         number of timesteps and each row contains [theta, w]  
            init_traj_controls (np.ndarray): Initial trajectory controls with shape (N, 1) where
                                           each element is the torque input u
            display_flag (bool, optional): Flag to display convergence plots and pendulum animation.
                                         Defaults to False.
        
        Returns:
            tuple: A tuple containing:
                - X (np.ndarray): Optimized state trajectory in alternating format
                                  [theta_0, w_0, theta_1, w_1, ...] with shape (N*2, 1)  
                - U (np.ndarray): Optimized control trajectory with shape (N, 1)
        
        Notes:
            - Uses direct KKT system solution (no inequality constraints)
            - Implements filter line search for step size selection
            - Initial conditions are enforced to be zero (theta_0=0, w_0=0)
            - Displays convergence plots and animation if display_flag=True
            - Terminates when constraint violation and KKT conditions are satisfied
        """
        max_iter = 20                                             # max iterations of SQP
        stop_tol = 1e-5                                           # stop tolerance
        curr_iter = 0
        N = init_traj_states.shape[0]                             # number of timesteps
        num_vars = (N-1)*(self.conf.nx+self.conf.nu)+self.conf.nx # number of trajectory variables


        # Track constraint violation, running cost, and alpha per iteration of the solver
        constr_viol_list  = np.empty((max_iter, 1))
        running_cost_list = np.empty((max_iter, 1))
        alpha_list        = np.empty((max_iter, 1))

        # Create combined trajectory array, interleaving states and controls:
        # [theta_0, w_0, u_0, theta_1, w_1, u_1, ...]
        x_guess = np.zeros((num_vars, 1))
        for i in range(N - 1):
            start_idx = i * (self.conf.nx + self.conf.nu)
            x_guess[start_idx:start_idx + self.conf.nx, 0] = init_traj_states[i]
            x_guess[start_idx + self.conf.nx, 0] = init_traj_controls[i, 0]
        x_guess[-2:, 0] = init_traj_states[-1]

        # Ensure initial conditions are zero (theta_0 = 0, w_0 = 0) to help with convergence
        x_guess[0, 0] = 0.0  # theta_0
        x_guess[1, 0] = 0.0  # w_0
        
        lambda_guess = np.zeros(((self.conf.nx)*N, 1))
        mu_guess = np.zeros(((self.conf.nx)*N, 1))

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
            
            p_sol, H, grad_f, grad_g, g = self.env.construct_KKT_n_solve(x_guess)
            
            x_guess_dir = p_sol[:num_vars]
            lambdas_guess_dir = p_sol[num_vars:]
            
            # Filter line search
            # reset alpha and flag for each iteration
            alpha = 1
            accept_step = False

            while not accept_step:
                if alpha < 1e-5:
                    break
                # NOTE: (x_guess + alpha * x_guess_dir) has shape ((self.nu+self.nx)*self.N, 1)
                if np.linalg.norm(self.env.running_cost(x_guess + alpha * x_guess_dir)) < f_best:
                    f_best = self.env.running_cost(x_guess + alpha * x_guess_dir)
                    accept_step = True
                if self.env.get_amnt_constr_violation(x_guess + alpha * x_guess_dir) < c_best:
                    c_best = abs(self.env.get_amnt_constr_violation(x_guess + alpha * x_guess_dir))
                    accept_step = True
                else:
                    alpha = alpha * rho

            # Update guesses
            x_guess = x_guess + alpha*x_guess_dir
            lambda_guess = (1-alpha)*lambda_guess + alpha*lambdas_guess_dir

            # Record constraint violation, running cost, alpha per iteration
            constr_viol_list[curr_iter]  = self.env.get_amnt_constr_violation(x_guess)
            running_cost_list[curr_iter] = np.linalg.norm(self.env.running_cost(x_guess)) #f_best
            alpha_list[curr_iter]        = alpha

            KKT = grad_f.squeeze() + lambdas_guess_dir.T @ grad_g
            print("Curr iter: ", curr_iter, " Cost: ", running_cost_list[curr_iter],
                " Constraint Violation: ", constr_viol_list[curr_iter],
                "KKT: ", np.linalg.norm(KKT))
            
            # Move onto next iteration
            curr_iter += 1

        print("Total iterations: ", curr_iter)

        # Extract values of the pendulum system
        pend_thetas = x_guess[0::3][:N]
        pend_ws     = x_guess[1::3][:N]
        pend_us     = x_guess[2::3]   # Already N−1

        # Extract X and U from pend_states and pend_us
        pend_states = np.zeros((N, 3))
        for i in range(N):
            pend_states[i, 0] = pend_thetas[i, 0]
            pend_states[i, 1] = pend_ws[i, 0]
            pend_states[i, 2] = i * self.conf.dt 
        X = pend_states
        U = pend_us[:-1, :]

        if display_flag:
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
            plt.plot(np.arange(N), pend_thetas, label="theta (angle)")
            plt.ylabel('theta (angle)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(N), pend_ws, label="w (angular velocity)")
            plt.ylabel('w (angular velocity)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            plt.figure()
            plt.plot(np.arange(N-1), pend_us, label="u (control)")
            plt.ylabel('u (control signal)')
            plt.xlabel('timestep')
            plt.grid()
            plt.legend()

            x_init = np.array([[.0],
                               [.0]])
            pendulum.animate_robot(x_init, pend_us.T)

        return X, U