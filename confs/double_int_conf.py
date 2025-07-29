import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from confs.base_env import BaseEnv

#-----TO params------------------------------------------------------------------------------------
TO_EPISODES = 200                                                                                  # Number of episodes solving TO/computing reward before updating critic and actor
dt = 0.1                                                                                           # timestep
NSTEPS = 30                                                                                        # Max trajectory length
X_INIT_MIN = np.array([-1.0, -1.0, 0.0])                                                           # Initial position (x), velocity (v), timestep (t)
X_INIT_MAX = np.array([1.0, 1.0, (NSTEPS-1)*dt])                                                   # Final position (x), velocity (v), timestep (t)
nx = 2                                                                                             # Number of state variables
nq = 1                                                                                             # Number of joint positions
nu = 1                                                                                             # Number of actions (controls (torques for each joint)), other conventions use nu

#----- NN params-----------------------------------------------------------------------------------
NN_LOOPS = np.arange(100, 50000, 300)                                                              # Number of updates K of critic and actor performed every TO_EPISODES                                                                              
NN_LOOPS_TOTAL = 100000                                                                            # Max NNs updates total
BATCH_SIZE = 128                                                                                   # Num. of transitions sampled from buffer for each NN update
NH1 = 64                                                                                           # 1st hidden layer size - actor
NH2 = 64                                                                                           # 2nd hidden layer size - actor
NN_PATH = 'double_int'                                                                             # Path to save the .pth files for actor and critic
CRITIC_LEARNING_RATE = 5e-4                                                                        # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                         # Learning rate for the policy network
NORMALIZE_INPUTS = 0                                                                               # Flag to normalize inputs (state)
NORM_ARR = np.array([10,10,int(NSTEPS*dt)])                                                        # Array of values to normalize by
MAX_NORM_A = 1.0                                                                                   # Maximum norm of gradient for actor
MAX_NORM_C = 1.0                                                                                   # Maximum norm of gradient for critic

#-----Misc params----------------------------------------------------------------------------------
REPLAY_SIZE = 2**20                                                                                # Size of the replay buffer
MC = 0                                                                                             # Flag to use MC or TD(n)
UPDATE_RATE = 0.001                                                                                # Homotopy rate to update the target critic network if TD(n) is used
NSTEPS_TD_N = int(NSTEPS/2)
scale = 1e-3                                                                                       # Reward function scale
gamma = 0.9

#-----Double Integrator-specific params------------------------------------------------------------
goal_state = np.array([0.0, 0.0])                                                                  # Desired goal state (θ, w)
u_min = -5.0
u_max = 5.0
bound_NN_action = True

#-----Double Integrator Env & SQP Solver-----------------------------------------------------------
class DoubleIntegratorEnv(BaseEnv):
    def __init__(self, conf):
        super().__init__(conf)
        self.goal_state = conf.goal_state
        self.scale = conf.scale
        self.nx, self.nu = conf.nx, conf.nu
        self.dt = conf.dt
        self.num_eq_constraints = 2
        self.w_pos = 20.0
        self.w_vel = 2.0
        self.w_u = 0.1
        self.w_pos_f = 20.0
        self.w_vel_f = 2.0

    def running_cost(self, x):
        nx, nu = self.nx, self.nu
        x_g, v_g = self.goal_state
        N = (x.shape[0] - nx) // (nx + nu) + 1

        pos = x[0::nx+nu][:N]
        vel = x[1::nx+nu][:N]
        u   = x[2::nx+nu][:N-1]

        cost = np.sum(
            self.w_pos * (pos[:-1] - x_g)**2 +
            self.w_vel * (vel[:-1] - v_g)**2 +
            self.w_u  * u**2
        )
        cost += self.w_pos_f * (pos[-1] - x_g)**2 + self.w_vel_f * (vel[-1] - v_g)**2
        return cost

    def grad_running_cost(self, x):
        nx, nu = self.nx, self.nu
        x_g, v_g = self.goal_state
        N = (x.shape[0] - nx) // (nx + nu) + 1

        grad = np.zeros_like(x)
        for i in range(N - 1):
            idx = i * (nx + nu)
            grad[idx]     = 2 * self.w_pos * (x[idx] - x_g)
            grad[idx + 1] = 2 * self.w_vel * (x[idx + 1] - v_g)
            grad[idx + 2] = 2 * self.w_u  * x[idx + 2]
        # Terminal state
        idx = (N - 1) * (nx + nu)
        grad[idx]     = 2 * self.w_pos_f * (x[idx] - x_g)
        grad[idx + 1] = 2 * self.w_vel_f * (x[idx + 1] - v_g)
        return grad

    def hess_running_cost(self, x):
        nx, nu = self.nx, self.nu
        N = (x.shape[0] - nx) // (nx + nu) + 1
        num_vars = (N - 1) * (nx + nu) + nx
        hess = np.zeros((num_vars, num_vars))

        # Define the block-diagonal Hessian for each time step
        H_i = 2* np.array([[self.w_pos,  0,  0],
                        [ 0, self.w_vel,  0],
                        [ 0,  0, self.w_u]])
        
        # Fill the block-diagonal Hessian matrix
        for i in range(N - 1):
            # Place H_i into the Hessian matrix at the appropriate location
            start = i * (nx + nu)
            end = start + nx + nu
            hess[start:end, start:end] = H_i
        start = (N - 1) * (nx + nu)
        H_last = 2 * np.array([[self.w_pos_f, 0], [0, self.w_vel_f]])
        hess[start:start+nx, start:start+nx] = H_last
        return hess

    def get_linearized_constraints(self, x):
        nx, nu, dt = self.nx, self.nu, self.dt
        N = (x.shape[0] - nx) // (nx + nu) + 1

        pos = x[0::nx+nu][:N]
        vel = x[1::nx+nu][:N]
        u   = x[2::nx+nu][:N-1]

        g0 = np.array([[0.], [0.]]).flatten()
        g_dyn_pos = (pos[1:] - pos[:-1] - dt * vel[:-1]).flatten()
        g_dyn_vel = (vel[1:] - vel[:-1] - dt * u).flatten()

        g_dyn = np.empty(2*(N-1))
        g_dyn[0::2], g_dyn[1::2] = g_dyn_pos, g_dyn_vel
        g = np.concatenate([g0, g_dyn]).reshape(-1, 1)
        return g

    def get_grad_linearized_constraints(self, x):
        nx, nu, dt = self.nx, self.nu, self.dt
        N = (x.shape[0] - nx) // (nx + nu) + 1
        num_vars = (N - 1) * (nx + nu) + nx

        G = np.zeros((self.num_eq_constraints * N, num_vars))
        I = np.eye(nx)

        # Initial constraint
        G[:nx, :nx] = I

        A = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        B = np.array([
            [0.0],
            [dt]
        ])

        for i in range(1, N):
            row_start = 2 * i
            col_xn     = (i - 1) * (nx + nu)
            col_un     = col_xn + nx
            col_xnp1   = i * (nx + nu)

            G[row_start:row_start+2, col_xn:col_xn+2] = A
            G[row_start:row_start+2, col_un:col_un+1] = B
            G[row_start:row_start+2, col_xnp1:col_xnp1+2] = -I

        return G

    def construct_KKT_n_solve(self, x):
        nx, nu = self.nx, self.nu
        N = (x.shape[0] - nx) // (nx + nu) + 1
        num_vars = (N - 1) * (nx + nu) + nx

        grad_f = self.grad_running_cost(x)
        H = self.hess_running_cost(x)
        g = self.get_linearized_constraints(x)
        grad_g = self.get_grad_linearized_constraints(x)

        num_eq = grad_g.shape[0]

        KKT_mat = np.zeros((num_vars + num_eq, num_vars + num_eq))
        KKT_mat[:num_vars, :num_vars] = H
        KKT_mat[num_vars:, :num_vars] = grad_g
        KKT_mat[:num_vars, num_vars:] = grad_g.T

        rhs = np.vstack([-grad_f, g])
        p_sol = np.linalg.solve(KKT_mat, rhs)
        return p_sol, H, grad_f, grad_g, g

    def get_amnt_constr_violation(self, x):
        nx, nu, dt = self.nx, self.nu, self.dt
        N = (x.shape[0] - nx) // (nx + nu) + 1

        pos = x[0::nx+nu][:N]
        vel = x[1::nx+nu][:N]
        u   = x[2::nx+nu][:N-1]

        g = []
        for i in range(1, N):
            g.append(pos[i-1] + dt * vel[i-1] - pos[i])
            g.append(vel[i-1] + dt * u[i-1] - vel[i])
        g = np.array(g).reshape(-1, 1)
        return np.sum(np.abs(g))

    def reset_batch(self, batch_size):
        times = np.random.uniform(self.conf.X_INIT_MIN[-1], self.conf.X_INIT_MAX[-1], batch_size)
        states = np.random.uniform(
            self.conf.X_INIT_MIN[:-1], self.conf.X_INIT_MAX[:-1],
            size=(batch_size, len(self.conf.X_INIT_MAX[:-1]))
        )
        times_int = np.expand_dims(self.conf.dt * np.round(times / self.conf.dt), axis=1)
        return np.hstack((states, times_int))

    def simulate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Args:
            state (np.ndarray): shape (3,), [p, v, t]
            action (np.ndarray): shape (1,)

        Returns:
            np.ndarray: next state, shape (3,)
        """
        dt = self.conf.dt

        p, v, t = state[0], state[1], state[2]
        u = action[0]

        p_next = p + v * dt
        v_next = v + u * dt
        t_next = t + dt

        return np.array([p_next, v_next, t_next])


    def simulate_batch(self, state, action):
        """
        Args:
            state: torch.Tensor (batch_size, 3)
            action: torch.Tensor (batch_size, 1)
        Returns:
            next_state: torch.Tensor (batch_size, 3)
        """
        dt = self.conf.dt
        p, v, t = state[:, 0], state[:, 1], state[:, 2]
        u = action[:, 0]

        p_next = p + v * dt
        v_next = v + u * dt
        t_next = t + dt

        return torch.stack((p_next, v_next, t_next), dim=1)

    def derivative_batch(self, state, action):
        action = action.clone().detach().requires_grad_(True)
        next_states = self.simulate_batch(state, action)

        # Compute gradient of next_states w.r.t actions
        jacobian = []
        for i in range(next_states.shape[1]):
            grad_i = torch.autograd.grad(
                next_states[:, i],
                action,
                grad_outputs=torch.ones_like(next_states[:, i]),
                retain_graph=True,
                create_graph=True
            )[0]
            jacobian.append(grad_i.unsqueeze(1))
        return torch.cat(jacobian, dim=1)

    def reward(self, state, action=None):
        """
        Reward = negative squared distance to goal + control penalty.
        state: [p, v, t]
        """
        goal = self.goal_state
        cost = self.w_pos * (state[0] - goal[0])**2 + self.w_vel * (state[1] - goal[1])**2
        if action is not None:
            cost += self.w_u * action[0]**2
        cost *= self.scale
        return -cost

    def reward_batch(self, state_batch, action_batch=None):
        goal = self.goal_state
        cost = self.w_pos * (state_batch[:, 0] - goal[0])**2 + self.w_vel * (state_batch[:, 1] - goal[1])**2
        if action_batch is not None:
            cost += self.w_u * (action_batch[:, 0])**2
        cost *= self.scale
        return -cost.unsqueeze(1)