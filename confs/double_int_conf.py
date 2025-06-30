import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from confs.base_env import BaseEnv

#-----TO params------------------------------------------------------------------------------------
TO_EPISODES = 50                                                                                   # Number of episodes solving TO/computing reward before updating critic and actor
dt = 0.01                                                                                          # timestep
NSTEPS = 500                                                                                       # Max trajectory length
X_INIT_MIN = np.array([-1.0, -1.0, 0.0])                                                           # Initial position (x), velocity (v), timestep (t)
X_INIT_MAX = np.array([1.0, 1.0, (NSTEPS-1)*dt])                                                   # Final position (x), velocity (v), timestep (t)
nx = 2                                                                                             # Number of state variables
nq = 1                                                                                             # Number of joint positions
nu = 1                                                                                             # Number of actions (controls (torques for each joint)), other conventions use nu

#----- NN params-----------------------------------------------------------------------------------
NN_LOOPS = np.arange(100, 4800, 300)                                                               # Number of updates K of critic and actor performed every TO_EPISODES                                                                              
NN_LOOPS_TOTAL = 10000                                                                             # Max NNs updates total
BATCH_SIZE = 128                                                                                   # Num. of transitions sampled from buffer for each NN update
NH1 = 64                                                                                           # 1st hidden layer size - actor
NH2 = 64                                                                                           # 2nd hidden layer size - actor
NN_PATH = 'double_int'                                                                             # Path to save the .pth files for actor and critic
CRITIC_LEARNING_RATE = 5e-4                                                                        # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                         # Learning rate for the policy network
NORMALIZE_INPUTS = 0                                                                               # Flag to normalize inputs (state)
NORM_ARR = np.array([10,10,int(NSTEPS*dt)])                                                        # Array of values to normalize by

#-----Misc params----------------------------------------------------------------------------------
REPLAY_SIZE = 2**20                                                                                # Size of the replay buffer
MC = 0                                                                                             # Flag to use MC or TD(n)
UPDATE_RATE = 0.001                                                                                # Homotopy rate to update the target critic network if TD(n) is used
NSTEPS_TD_N = int(NSTEPS/4)
scale = 1e-3                                                                                       # Reward function scale

#-----Double Integrator-specific params------------------------------------------------------------
goal_state = np.array([0.0, 0.0])                                                                  # Desired goal state (θ, w)

#-----Double Integrator Env & SQP Solver-----------------------------------------------------------
class DoubleIntegratorEnv(BaseEnv):
    def __init__(self, conf):
        super().__init__(conf)
        self.goal_state = torch.tensor(conf.goal_state, dtype=torch.float32)
        self.scale = conf.scale

    def reset_batch(self, batch_size):
        times = np.random.uniform(self.conf.X_INIT_MIN[-1], self.conf.X_INIT_MAX[-1], batch_size)
        states = np.random.uniform(
            self.conf.X_INIT_MIN[:-1], self.conf.X_INIT_MAX[:-1],
            size=(batch_size, len(self.conf.X_INIT_MAX[:-1]))
        )
        times_int = np.expand_dims(self.conf.dt * np.round(times / self.conf.dt), axis=1)
        return np.hstack((states, times_int))

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
        goal = self.goal_state.to(state.device)
        cost = 10.0 * (state[0] - goal[0])**2 + 0.1 * (state[1] - goal[1])**2
        if action is not None:
            cost += 0.01 * action[0]**2
        cost *= self.scale
        return -cost

    def reward_batch(self, state_batch, action_batch=None):
        goal = self.goal_state.to(state_batch.device)
        cost = 10.0 * (state_batch[:, 0] - goal[0])**2 + 0.1 * (state_batch[:, 1] - goal[1])**2
        if action_batch is not None:
            cost += 0.01 * (action_batch[:, 0])**2
        cost *= self.scale
        return -cost.unsqueeze(1)