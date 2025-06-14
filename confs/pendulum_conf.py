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

#----- NN params-----------------------------------------------------------------------------------
NN_LOOPS = np.arange(1000, 48000, 3000)                                                            # Number of updates K of critic and actor performed every TO_EPISODES                                                                              
NN_LOOPS_TOTAL = 100000                                                                            # Max NNs updates total
BATCH_SIZE = 128                                                                                   # Num. of transitions sampled from buffer for each NN update
NH1 = 256                                                                                          # 1st hidden layer size - actor
NH2 = 256                                                                                          # 2nd hidden layer size - actor
NN_PATH = 'pendulum'                                                                               # Path to save the .pth files for actor and critic
CRITIC_LEARNING_RATE = 5e-4                                                                        # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                         # Learning rate for the policy network
kreg_l1_A = 1e-2                                                                                   # Weight of L1 regularization in actor's network - kernel
kreg_l2_A = 1e-2                                                                                   # Weight of L2 regularization in actor's network - kernel
breg_l1_A = 1e-2                                                                                   # Weight of L2 regularization in actor's network - bias
breg_l2_A = 1e-2                                                                                   # Weight of L2 regularization in actor's network - bias
kreg_l1_C = 1e-2                                                                                   # Weight of L1 regularization in critic's network - kernel
kreg_l2_C = 1e-2                                                                                   # Weight of L2 regularization in critic's network - kernel
breg_l1_C = 1e-2                                                                                   # Weight of L1 regularization in critic's network - bias
breg_l2_C = 1e-2                                                                                   # Weight of L2 regularization in critic's network - bias

#-----TO params------------------------------------------------------------------------------------
TO_EPISODES = 10                                                                                   # Number of episodes solving TO/computing reward before updating critic and actor
dt = pendulum.dt                                                                                   # timestep
NSTEPS = 300                                                                                       # Max trajectory length
X_INIT_MIN = np.array([0.0, 0.0, 0.0])                                                             # Initial angle (θ),  angular velocity (w), timestep (t)
X_INIT_MAX = np.array([np.pi, 0.0, (NSTEPS-1)*dt])                                                 # Final angle (θ),  angular velocity (w), timestep (t)
nx = 2                                                                                             # Number of state variables (7 joint positions + 7 joint velocities)
nq = 1                                                                                             # Number of joint positions (KUKA IIWA has 7 joints)
nu = 1                                                                                             # Number of actions (controls (torques for each joint)), other conventions use nu

#-----Misc params----------------------------------------------------------------------------------
REPLAY_SIZE = 2**16                                                                                # Size of the replay buffer
MC = 0                                                                                             # Flag to use MC or TD(n)
UPDATE_RATE = 0.001                                                                                # Homotopy rate to update the target critic network if TD(n) is used
NSTEPS_TD_N = int(NSTEPS/4)  
NORMALIZE_INPUTS = 0                                                                               # Flag to normalize inputs (state)
NORM_ARR = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10, int(NSTEPS*dt)])                   # Array of values to normalize by

#-----pendulum-specific params----------------------------------------------------------------------
goal_state = np.array([np.pi, 0.0])                                                                 # Desired goal state (θ, w)
u_min = 10
u_max = 10
g = pendulum.g
num_eq_constraints = 2

#-----Pendulum Env & SQP Solver--------------------------------------------------------------------
class PendulumEnv(BaseEnv):
    def __init__(self, conf, N_ts=NSTEPS, u_min=u_min, u_max=u_max):
        # NOTE: Passing the number of timsteps N from the environment initialization for flexibility
        # we might want to change it later depending on how the environment is initialized and called
        # in batches
        super().__init__(conf)
        self.conf = conf
        self.dt = dt                                                                                # Time step for the simulation
        self.g = g                                                                                  # gravity
        self.N = N_ts                                                                               # Number of time steps
        self.nq = nq                                                                                # Number of joints (1 for pendulum)
        self.nx = nx                                                                                # Number of state variables (1 joint position + 1 joint velocity)
        self.nu = nu                                                                                # Number of actuators (1 for pendulum torque)
        self.goal_state = goal_state                                                                # Target state (pendulum upright position)
        self.num_vars = (self.N - 1) * (nx + nu) + nx                                                         # Total number of variables in trajectory
        self.num_eq_constraints = num_eq_constraints                                                # Number of equality constraints
        self.u_min = u_min                                                                          # min control
        self.u_max = u_max                                                                          # max control

    def running_cost(self, x):
        """
        Compute the running cost for the pendulum system.
        The cost is defined as the squared difference from the target state.

        Args:
            x (np.ndarray): Trajectory of shape ((N-1)*(nx+nu)+nx,)
        Returns:
            float: Total running cost for the trajectory
        """
        theta = x[0::3] # angle of pendulum, extracted every third element from x starting from index 0
        w     = x[1::3] # angular velocity of pendulum that's extract every third element from x starting from index 1
        u     = x[2::3] # control input (torque) that's extracted every third element from x starting from index 2

        # running cost vector f
        f = []
        
        # Compute rest of the entries
        for i in range(0, self.N - 1):
            f.append(10*(theta[i]-self.goal_state[0])**2 + 0.1*(w[i]-self.goal_state[1])**2 + 0.1*u[i]**2)
        f.append(10*(theta[self.N-1]-self.goal_state[0])**2 + 0.1*(w[self.N-1]-self.goal_state[1])**2)

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
        grad = np.empty((self.num_vars, 1))
        i = 0 # index to go through
        for i in range(self.N - 1):
            idx = i * (self.nx + self.nu)
            grad[idx]     = 20 * (x[idx] - self.goal_state[0])      # 20(theta_i - pi)
            grad[idx + 1] = 0.2 * (x[idx + 1] - self.goal_state[1]) # 0.2(w_i - 0)
            grad[idx + 2] = 0.2 * x[idx + 2]                        # 0.2u_i
        idx = (self.N - 1) * (self.nx + self.nu)
        grad[idx]     = 20 * (x[idx] - self.goal_state[0])
        grad[idx + 1] = 0.2 * (x[idx + 1] - self.goal_state[1])
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
        for i in range(self.N - 1):
            # Calculate the starting index for the current block
            idx = i * (self.nx + self.nu)
            # Place H_i into the Hessian matrix at the appropriate location
            hess[idx:idx + self.nx + self.nu, idx:idx + self.nx + self.nu] = H_i
        idx = (self.N - 1) * (self.nx + self.nu)
        H_last = np.array([[20, 0], [0, 0.2]])
        hess[idx:idx + self.nx, idx:idx + self.nx] = H_last
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
            g.append(-w[i-1] - dt*u[i-1] + dt*self.g*np.sin(theta[i-1]) + w[i])

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
        I = np.eye(self.nx)
        G[0:self.nx, 0:self.nx] = I

        # Construct B, note that A changes every time it's called
        B = np.array([[0.],
                    [dt]])

        # Construct the G matrix
        for i in range(1, self.N):
            # Construct A for every changing theta_n
            A = np.array([[             1.,              self.dt],
                        [-dt*self.g*np.cos(theta[i-1][0]), 1.]])

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
            g.append(w[i-1] + dt*u[i-1] - dt*self.g*np.sin(theta[i-1]) - w[i])

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
        N = self.N
        nx, nu = self.nx, self.nu
        num_vars = (N-1)*(nx + nu) + nx

        # NOTE: First construct H (which is constant, hence why we make it once out here to save time)
        # h_n matrix that makes up the big H matrix
        h_n = np.array([[0, 0,  1],
                        [0, 0, -1]])
        p = 2 # since upper and lower bounds on control u
        H = np.zeros((p * (N - 1), num_vars))

        # Loop over for each block
        for n in range(N-1):
            # Cacluate row and col indices for current block
            row_start = 2 * n
            row_end = row_start + 2
            col_start = (nx + nu) * n
            col_end = col_start + (nx + nu)

            # Place h_n at diagonals for H
            H[row_start:row_end, col_start:col_end] = h_n

        # NOTE: Now construct h
        # extract all the u's from x_guess
        u = x[2::(nx + nu)].flatten()[:N - 1]
        
        # interweave the u_max - u_n and u_min + u_n
        h = np.empty(2 * (N - 1))
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
            g.append(w[i-1] + dt*u[i-1] - dt*self.g*np.sin(theta[i-1]) - w[i])

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
            batch_size (int): Number of initial states to reset

        Returns:
            np.ndarray: Array of shape (batch_size, 3), where each row is the state
                        appended by the timestep
        """
        times = np.random.uniform(self.conf.X_INIT_MIN[-1], self.conf.X_INIT_MAX[-1], batch_size)
        states = np.random.uniform(
            self.conf.X_INIT_MIN[:-1], self.conf.X_INIT_MAX[:-1],
            size=(batch_size, len(self.conf.X_INIT_MAX[:-1]))
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
        state, action = torch.tensor(state), torch.tensor(action)
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