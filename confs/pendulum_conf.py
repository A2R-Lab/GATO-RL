import numpy as np
import matplotlib.pyplot as plt
import pendulum

from qpsolvers import solve_qp, Problem, solve_problem

"""
Trajectory optimization class for a simple pendulum system.

The class itself is an SQP (sequential quadratic programming) solver that optimizes the
trajectory of a pendulum system. The pendulum Python file contains the animation function
to visualize the pendulum's motion based on the control trajectory and initial position.

For an SQP solver, the dynamics is respected by the equality constraints. Inequality constraints
are used to limit the control inputs and the state variables.

NOTE: For our purposes, we use the unconstrained version for speed.
NOTE: The pendulum when down is at theta = 0, and when up is at theta = pi.

The following is an example of the pendulum simulation and animation without control inputs

controls = np.zeros((N, 1))  # Control inputs (torque)
x_init = np.array([[1.0], [0.0]])  # Initial state (angle, angular velocity)

pendulum.animate_robot(x_init, controls.T)
"""

#-----CACTO params---------------------------------------------------------------------------------
# TODO: fill out CACTO parameters for pendulum case

#----- NN params-----------------------------------------------------------------------------------
# TODO: fill out NN parameters for pendulum case

#-----TO params------------------------------------------------------------------------------------
dt = pendulum.dt
N = 300 # Number of time steps

# Dimension of the state vector
x_dim = 2  # [theta (angle), w (angular velocity)]
# Dimension of the control
u_dim = 1  # [torque]

# Total number of variables
num_vars = N * (x_dim + u_dim)  # Total number of variables in trajectory

# Number of equality constraints
num_eq_constraints = 2 # 

# Gravity constant
grav = pendulum.g

#-----Pendulum Env & SQP Solver--------------------------------------------------------------------
class PendulumEnv:
    def __init__(self, conf):
        self.conf = conf
        self.dt = dt # Time step for the simulation
        self.N = N # Number of time steps
        self.nq = 1 # Number of joints (1 for pendulum)
        self.nx = x_dim # Number of state variables (1 joint position + 1 joint velocity)
        self.nu = u_dim # Number of actuators (1 for pendulum torque)
        self.TARGET_STATE = np.array([[np.pi], [0.0]])  # Target state (pendulum upright position)
        self.num_vars = self.N * (self.nx + self.nu) # Total number of variables in trajectory
        self.num_eq_constraints = num_eq_constraints  # Number of equality constraints

    def running_cost(self, x):
        """
        Compute the running cost for the pendulum system.
        The cost is defined as the squared difference from the target state.

        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            float: Total running cost for the trajectory.
        """
        theta = x[0::3] # angle of pendulum, extracted every third element from x starting from index 0
        w     = x[1::3] # angular velocity of pendulum that's extract every third element from x starting from index 1
        u     = x[2::3] # control input (torque) that's extracted every third element from x starting from index 2

        # running cost vector f
        f = []
        
        # Compute rest of the entries
        for i in range(0, self.N):
            f.append(10*(theta[i]-np.pi)**2 + 0.1*w[i]**2 + 0.1*u[i]**2)
        
        # convert list to numpy vector
        f = np.array(f).reshape(-1, 1)
        return np.sum(f)
    
    def grad_running_cost(self, x):
        """
        Get the gradient of the running cost from the given trajectory:
        x = [theta_0, w_0, u_0, ..., theta_N, w_N, u_N]^T
        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: Gradient of the running cost with respect to the trajectory variables.
        """
        grad = np.empty(((self.nx + self.nu)*self.N, 1))
        i = 0 # index to go through
        while i < self.num_vars:
            grad[i] = 20 * (x[i] - np.pi) # 20(theta_i - pi)
            grad[i+1] = 0.2 * x[i+1]      # 0.2w_i
            grad[i+2] = 0.2 * x[i+2]      # 0.2u_i
            i += x_dim + u_dim
        return grad
    
    def hess_running_cost(self, x):
        """
        Get the Hessian of the running cost from the given trajectory:
        x = [theta_0, w_0, u_0, ..., theta_N, w_N, u_N]^T
        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: Hessian of the running cost with respect to the trajectory variables.
        """
        hess = np.zeros((self.num_vars, self.num_vars))  # Initialize Hessian matrix
        # Define the block-diagonal Hessian for each time step
        H_i = np.array([[20,  0,  0],
                        [ 0, .2,  0],
                        [ 0,  0, .2]])
        
        # Fill the block-diagonal Hessian matrix
        for i in range(self.N):
            # Calculate the starting index for the current block
            idx = i * (x_dim + u_dim)
            # Place H_i into the Hessian matrix at the appropriate location
            hess[idx:idx + x_dim + u_dim, idx:idx + x_dim + u_dim] = H_i

        return hess

    def get_linearized_constraints(self, x):
        """
        Get the linearized constraints of the equality constraints from the given trajectory:
        x = [theta_0, w_0, u_0, ..., theta_N, w_N, u_N]^T
        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: Linearized constraints vector g.
        """
        theta = x[0::3] # extract every third element from x starting from index 0
        w     = x[1::3]
        u     = x[2::3]

        # constraint vector g
        g = []

        # Add first two elements, which are our initial states (zeros)
        g.append(np.array([0]))
        g.append(np.array([0]))

        # Compute rest of the entries
        for i in range(1, self.N):
            # Two appended due to two equality constraints
            g.append(-theta[i-1] - dt*w[i-1] + theta[i])
            g.append(-w[i-1] - dt*u[i-1] + dt*grav*np.sin(theta[i-1]) + w[i])

        # convert list to numpy vector
        g = np.array(g).reshape(-1, 1)
        return g
    
    def get_grad_linearized_constraints(self, x):
        """
        Build up the matrices used to construct the equality constraint matrix
        """
        # Intialize the gradient matrix G
        G = np.zeros((self.num_eq_constraints*self.N, self.num_vars))
        theta = x[0::3] # extract every third element from x starting from index 0

        # Add first identity matrix in G
        I = np.eye(x_dim)
        G[0:x_dim, 0:x_dim] = I

        # Construct B, note that A changes every time it's called
        B = np.array([[0.],
                    [dt]])

        # Construct the G matrix
        for i in range(1, self.N):
            # Construct A for every changing theta_n
            A = np.array([[             1.,              self.dt],
                        [-dt*grav*np.cos(theta[i-1][0]), 1.]])

            # Row indices for the current block
            row_start = 2 * i
            row_end = row_start + 2

            # Column indices for x_n (state at time n)
            col_xn_start = 3 * (i - 1)
            col_xn_end = col_xn_start + 2

            # Column indices for u_n (control at time n)
            col_un = col_xn_end  # Since u_n follows x_n
            col_un_end = col_un + 1

            # Column indices for x_{n+1} (state at time n+1)
            col_xnp1_start = 3 * i
            col_xnp1_end = col_xnp1_start + 2

            # Place A in G
            G[row_start:row_end, col_xn_start:col_xn_end] = A

            # Place B in G
            G[row_start:row_end, col_un:col_un_end] = B

            # Place -I in G
            G[row_start:row_end, col_xnp1_start:col_xnp1_end] = -I
        
        return G
    
    def construct_KKT_n_solve(self, x):
        """
        Construct the Inner Linear KKT System

        NOTE:
        p_sol = [p_k, p_lambda]^T
        """
        grad_f = self.grad_running_cost(x)
        H = self.hess_running_cost(x)
        g = self.get_linearized_constraints(x)
        grad_g = self.get_grad_linearized_constraints(x)

        num_constr_grad_g = grad_g.shape[0]

        KKT_mat = np.zeros((self.num_vars + num_constr_grad_g,
                            self.num_vars + num_constr_grad_g))
        KKT_mat[0:self.num_vars, 0:self.num_vars] = H
        KKT_mat[self.num_vars:, 0:self.num_vars]  = grad_g
        KKT_mat[0:self.num_vars, self.num_vars:]  = grad_g.T

        p_sol = np.linalg.solve(KKT_mat, np.vstack((-grad_f, g)))

        return p_sol, H, grad_f, grad_g, g
    
    def get_amnt_constr_violation(self, x):
        """
        Compute the sum of the absolute values of the equality constraints
        NOTE: x = [theta_0, w_0, u_0, ..., theta_N, w_N, u_N]^T
        """
        theta = x[0::3] # extract every third element from x starting from index 0
        w     = x[1::3]
        u     = x[2::3]

        # constraint vector g
        g = []

        # Compute rest of the entries
        for i in range(1, self.N):
            # Two appended due to two equality constraints
            g.append(theta[i-1] + dt*w[i-1] - theta[i])
            g.append(w[i-1] + dt*u[i-1] - dt*grav*np.sin(theta[i-1]) - w[i])

        # convert list to numpy vector
        g = np.array(g).reshape(-1, 1)

        # get absolute value sum of g
        constr_violation = np.sum(np.abs(g))

        return constr_violation
    
    def solve_unconstrained_SQP(self):
        """
        Main Loop for SQP without Inequality Constraints
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
                # NOTE: (x_guess + alpha * x_guess_dir) has shape ((u_dim+x_dim)*N_timesteps, 1)
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

        print(x_guess.shape)


if __name__ == "__main__":
    """
    # Animate the pendulum with no control inputs
    controls = np.zeros((N, 1))  # Control inputs (torque)
    x_init = np.array([[1.0], [0.0]])  # Initial state (angle, angular velocity)
    
    pendulum.animate_robot(x_init, controls.T)
    """
    
    # Create pendulum environment and solve SQP system
    print("Creating PendulumEnv and solving SQP system...")
    pend_env = PendulumEnv(None)  # conf parameter not used in this implementation
    pend_env.solve_unconstrained_SQP()
    
    # Show all plots
    plt.show()