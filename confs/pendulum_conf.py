import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

from confs import pendulum

from confs.base_env import BaseEnv

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
dt = pendulum.dt                                                                                    # dt=0.01
N = 300                                                                                             # Number of time steps, i.e. max episode length
x_init_min = np.array([0.0, 0.0, 0.0])                                                              # Initial angle (θ),  angular velocity (w), timestep (t)
x_init_max = np.array([np.pi, 0.0, N * dt])                                                         # Final angle (θ),  angular velocity (w), timestep (t)
goal_state = np.array([np.pi, 0.0])                                                                 # Desired goal state (θ, w)

x_dim = 2                                                                                           # Dimension of the state vector [theta (angle), w (angular velocity)]
u_dim = 1                                                                                           # Dimension of the control [torque]

nx = x_dim                                                                                          # Number of state variables 
nq = 1                                                                                              # Number of joints (1 for pendulum)
nu = u_dim                                                                                          # Number of actuators (1 for pendulum torque)

num_vars = N * (x_dim + u_dim)                                                                      # Total number of variables in trajectory
num_eq_constraints = 2                                                                              # Number of equality constraints

grav = pendulum.g                                                                                   # Gravity constant

#-----Pendulum Env & SQP Solver--------------------------------------------------------------------
class PendulumEnv(BaseEnv):
    def __init__(self, conf, N_ts, u_min=10, u_max=10):
        # NOTE: Passing the number of timsteps N from the environment initialization for flexibility
        # we might want to change it later depending on how the environment is initialized and called
        # in batches
        super().__init__(conf)
        self.conf = conf
        self.dt = dt # Time step for the simulation
        self.g = grav
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
            p_sol = solve_problem(problem, solver="osqp")
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

    def reset_batch(self, batch_size):
        """
        Reset the environment to a random initial state for a batch of size `batch_size`.

        Args:
            batch_size (int): Number of initial states to reset

        Returns:
            np.ndarray: Array of shape (batch_size, 3), where each row is the state
                        appended by the timestep
        """
        times = np.random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1], batch_size)
        states = np.random.uniform(
            self.conf.x_init_min[:-1], self.conf.x_init_max[:-1],
            size=(batch_size, len(self.conf.x_init_max[:-1]))
        )
        times_int = np.expand_dims(self.conf.dt * np.round(times / self.conf.dt), axis=1)
        return np.hstack((states, times_int))

    def simulate(self, state, action):
        """
        Simulates one step forward using:
            θ_{t+1} = θ_t + dt * θ̇_t
            θ̇_{t+1} = θ̇_t + dt * (u_t - g * sin(θ_t))

        Args:
            state (np.ndarray): State with shape (3,) [θ_t, θ̇_t, t]
            action (np.ndarray): Action with shape (1,) [u_t]
        
        Returns:
            np.ndarray: Next state with shape (3,) [θ_{t+1}, θ̇_{t+1}, t+1]
        """
        theta_next = state[0] + self.dt * state[1]
        theta_dot_next = state[1] + self.dt * (action[0] - self.g * np.sin(state[0]))
        t_next = state[2] + self.dt
        return np.array([theta_next, theta_dot_next, t_next])

    def simulate_batch(self, state, action):
        """
        Batch version of simulate().

        References:
            neural_network.compute_actor_grad()

        Args:
            state (np.ndarray): Batch of states with shape (batch_size, 3)
            action (np.ndarray): Batch of actions with shape (batch_size, 1)

        Returns:
            torch.Tensor: Batch of next states with shape (batch_size, 3)
        """
        theta, theta_dot, t = state[:, 0], state[:, 1], state[:, 2]
        u = action[:, 0]

        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * (u - self.g * torch.sin(theta))
        t_next = t + self.dt
        return torch.stack([theta_next, theta_dot_next, t_next], dim=1)

    def derivative_batch(self, state, action):
        """
        Batch version of derivative(). Since only the angular velocity (ω) is affected by the control,
        the Jacobian dnext_state/daction has only one nonzero entry, ∂ω/∂u = dt.

        References:
            neural_network.compute_actor_grad()

        Args:
            state (np.ndarray): Batch of states (batch_size, 3)
            action (np.ndarray): Batch of actions (batch_size, 1)

        Returns:
            torch.Tensor: Batch of derivative matrices (batch_size, 3, 1)
        """
        batch_size = state.shape[0]
        dt = self.conf.dt
        jac = np.zeros((batch_size, 3, 1))
        jac[:, 1, 0] = dt
        return torch.tensor(jac, dtype=torch.float32)

    def reward(self, state, action=None):
        """
        Computes reward from state and action, penalizing deviation from goal:
            cost = 10 * (θ - θ*)² + 0.1 * (θ̇ - θ̇*)² + 0.1 * u²

        Args:
            state (np.ndarray): Current state
            action (np.ndarray, optional): Control input

        Returns:
            float: Reward value
        """
        goal_theta, goal_theta_dot = self.goal_state[:2]
        
        cost = 10.0 * (state[0] - goal_theta)**2 + 0.1 * (state[1] - goal_theta_dot)**2
        if action is not None:
            cost += 0.1 * action[0]**2
        return -cost

    def reward_batch(self, state_batch, action_batch=None):
        """
        Computes batch of rewards using tensors, penalizing deviation from goal.

        Args:
            state_batch (torch.Tensor): (batch_size, 3)
            action_batch (torch.Tensor, optional): (batch_size, 1)

        Returns:
            torch.Tensor: (batch_size, 1) reward values
        """
        theta, theta_dot = state_batch[:, 0], state_batch[:, 1]
        goal_theta, goal_theta_dot = self.goal_state[0], self.goal_state[1]
        
        cost = 10.0 * (theta - goal_theta)**2 + 0.1 * (theta_dot - goal_theta_dot)**2
        if action_batch is not None:
            cost += 0.1 * action_batch[:, 0]**2
        return -cost.unsqueeze(1)