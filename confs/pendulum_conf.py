import numpy as np
import matplotlib.pyplot as plt
import pendulum
from base_env import BaseEnv

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
dt = pendulum.dt # dt=0.01
N = 300 # Number of time steps, i.e. max episode length

# Min & max values for initial state vector + timestep
x_init_min = np.array([[0.0],  # Initial angle (theta)
                       [0.0],  # Initial angular velocity (w)
                       [0.0]]) # Initial timestep

# Final state vector + timestep
x_init_max = np.array([[np.pi], # Final angle (theta)
                       [0.0],   # Final angular velocity (w)
                       [N*dt]]) # Final timestep

# Desired goal state for the pendulum
goal_state = np.array([[np.pi],  # Desired angle (theta)
                       [0.0]])   # Desired angular velocity (w)

# Dimension of the state vector
x_dim = 2  # [theta (angle), w (angular velocity)]
# Dimension of the control
u_dim = 1  # [torque]

nx = x_dim # Number of state variables 
nq = 1  # Number of joints (1 for pendulum)
nu = u_dim  # Number of actuators (1 for pendulum torque)

# Total number of variables
num_vars = N * (x_dim + u_dim)  # Total number of variables in trajectory

# Number of equality constraints
num_eq_constraints = 2 # 

# Gravity constant
grav = pendulum.g

#-----Pendulum Env & SQP Solver--------------------------------------------------------------------
class PendulumEnv(BaseEnv):
    def __init__(self, conf, N_ts, u_min=10, u_max=10):
        # NOTE: Passing the number of timsteps N from the environment initialization for flexibility
        # we might want to change it later depending on how the environment is initialized and called
        # in batches
        super().__init__(conf)
        self.conf = conf
        self.dt = dt # Time step for the simulation
        self.N = N_ts # Number of time steps
        self.nq = 1 # Number of joints (1 for pendulum)
        self.nx = x_dim # Number of state variables (1 joint position + 1 joint velocity)
        self.nu = u_dim # Number of actuators (1 for pendulum torque)
        self.goal_state = goal_state  # Target state (pendulum upright position)
        self.num_vars = self.N * (self.nx + self.nu) # Total number of variables in trajectory
        self.num_eq_constraints = num_eq_constraints  # Number of equality constraints
        self.u_min = u_min
        self.u_max = u_max

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
            f.append(10*(theta[i]-self.goal_state[0])**2 + 0.1*(w[i]-self.goal_state[1])**2 + 0.1*u[i]**2)
        
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
            grad[i] = 20 * (x[i] - self.goal_state[0]) # 20(theta_i - pi)
            grad[i+1] = 0.2 * (x[i+1] - self.goal_state[1])      # 0.2(w_i - 0)
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

        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: Linearized constraints matrix G.
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
        which is both the solution to the KKT system
        and the Lagrange multipliers

        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: p_sol solution to the KKT system and the Lagrange multipliers.
            np.ndarray: H Hessian of the running cost.
            np.ndarray: grad_f Gradient of the running cost.
            np.ndarray: grad_g Gradient of the linearized constraints.
            np.ndarray: g Linearized constraints vector.
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
        Compute the sum of the absolute values of just the equality constraints

        The goal is to drive this down to zero!

        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: Sum of the absolute values of the equality constraints.
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
    
    def create_ineq_constraints(self, x):
        """
        Creating inequality bounds on the control u_min <= u_n <= u_max

        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: matrix H for the inequality constraints.
            np.ndarray: vector h for the inequality constraints.
        """

        # NOTE: First construct H (which is constant, hence why we make it once out here to save time)
        # h_n matrix that makes up the big H matrix
        h_n = np.array([[0, 0,  1],
                        [0, 0, -1]])
        p = 2 # since upper and lower bounds on control u
        H = np.zeros((p*self.N, (x_dim+u_dim)*self.N))

        # Loop over for each block
        for n in range(self.N):
            # Cacluate row and col indices for current block
            row_start = 2*n
            row_end = row_start + 2
            col_start = 3*n
            col_end = col_start + 3

            # Place h_n at diagonals for H
            H[row_start:row_end, col_start:col_end] = h_n

        # NOTE: Now construct h
        # extract all the u's from x_guess
        u = x[2::3].flatten()
        
        # interweave the u_max - u_n and u_min + u_n
        h = np.empty(2*len(u))
        h[0::2] = self.u_max - u
        h[1::2] = self.u_min + u

        h = h.reshape(-1, 1)

        return H, h
    
    def get_amnt_constr_violation_update_2(self, x):
        """
        Compute the sum of the absolute values of the equality constraints

        But now with inequality constraints as well.

        Args:
            x (np.ndarray): Trajectory of shape (nx+nu, N+1) where nx is the number of state variables.
        Returns:
            np.ndarray: Total constraint violation, including both equality and inequality constraints.
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

        for i in range(len(u)):
            if u[i] > self.u_max:
                #print("u_max violated: ", u[i])
                constr_violation += abs(u[i] - self.u_max)
            elif u[i] < -self.u_min:
                constr_violation += abs(u[i] - self.u_min)
                #print("u_min violated: ", u[i])
            else:
                pass

        return constr_violation

    def reset_batch(self, batch_size):
        """
   		Reset the environment to a random initial state for a batch of size `batch_size`.

        Args:
            batch_size (int): Number of init states to reset
        Returns:
            np.ndarray: Each row is the state appended by the timestep? TODO: double check this
        """
        times = np.random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1], batch_size)
        states = np.random.uniform(
            self.conf.x_init_min[:-1], self.conf.x_init_max[:-1],
            size=(batch_size, len(self.conf.x_init_max[:-1]))
        )
        times_int = np.expand_dims(self.conf.dt * np.round(times / self.conf.dt), axis=1)
        return np.hstack((states, times_int))