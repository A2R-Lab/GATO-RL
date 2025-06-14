import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import torch
import os
import pinocchio as pin
from base_env import BaseEnv

#----- NN params-----------------------------------------------------------------------------------
NN_LOOPS = np.arange(1000, 48000, 3000)                                                            # Number of updates K of critic and actor performed every TO_EPISODES                                                                              
NN_LOOPS_TOTAL = 100000                                                                            # Max NNs updates total
BATCH_SIZE = 128                                                                                   # Num. of transitions sampled from buffer for each NN update
NH1 = 256                                                                                          # 1st hidden layer size - actor
NH2 = 256                                                                                          # 2nd hidden layer size - actor
NN_PATH = 'iiwa'                                                                                   # Path to save the .pth files for actor and critic
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
dt = 0.01                                                                                          # timestep
NSTEPS = 50                                                                                        # Max trajectory length
X_INIT_MIN = np.array([-2.967,-2.094,-2.967,-2.094,-2.967,-2.094,-3.054,                           # minimum initial state vector + time
                    1.57,1.57,1.57,1.57,1.57,1.57,1.57,0])
X_INIT_MAX = np.array([2.967,2.094,2.967,2.094,2.967,2.094,3.054,                                  # maximum initial state vector + time
                    1.57,1.57,1.57,1.57,1.57,1.57,1.57,(NSTEPS-1)*dt])
nx = 14                                                                                            # Number of state variables (7 joint positions + 7 joint velocities)
nq = 7                                                                                             # Number of joint positions (KUKA IIWA has 7 joints)
nu = 7                                                                                             # Number of actions (controls (torques for each joint)), other conventions use nu

#-----Misc params----------------------------------------------------------------------------------
REPLAY_SIZE = 2**16                                                                                # Size of the replay buffer
MC = 0                                                                                             # Flag to use MC or TD(n)
UPDATE_RATE = 0.001                                                                                # Homotopy rate to update the target critic network if TD(n) is used
NSTEPS_TD_N = int(NSTEPS/4)  
NORMALIZE_INPUTS = 0                                                                               # Flag to normalize inputs (state)
NORM_ARR = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10, int(NSTEPS*dt)])                   # Array of values to normalize by

#-----IIWA-specific params-------------------------------------------------------------------------
URDF_PATH = os.path.abspath(os.path.join('confs', 'iiwa.urdf'))
robot = RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[os.path.dirname(URDF_PATH)])
robot_data = robot.model.createData()
end_effector_frame_id = 'iiwa_link_7'
goal_ee = [0.5, 0.5, 0.5]

#-----env functions--------------------------------------------------------------------------------
class IiwaEnv(BaseEnv):
    def __init__(self, conf):
        super().__init__(conf)
        self.nx = conf.nx
        self.nq = conf.nq
        self.nu = conf.nu
        self.goal_ee = conf.goal_ee

    def reset_batch(self, batch_size):
        # NOTE: since the state vector has the time as the last element,
        # we generate random times and states separately, then concatenate them.
        times = np.random.uniform(self.conf.X_INIT_MIN[-1], self.conf.X_INIT_MAX[-1], batch_size)
        states = np.random.uniform(
            self.conf.X_INIT_MIN[:-1], self.conf.X_INIT_MAX[:-1],
            size=(batch_size, len(self.conf.X_INIT_MAX[:-1]))
        )
        times_int = np.expand_dims(self.conf.dt * np.round(times / self.conf.dt), axis=1)
        return np.hstack((states, times_int))

    def simulate(self, state, action):
        state_next = np.zeros(self.nx + 1)
        q, v = state[:self.nq], state[self.nq:self.nx]
        qdd = pin.aba(self.conf.robot.model, self.conf.robot_data, q, v, action)
        v_new = v + qdd * self.conf.dt
        q_new = pin.integrate(self.conf.robot.model, q, v_new * self.conf.dt)

        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(q_new), np.copy(v_new)
        state_next[-1] = state[-1] + self.conf.dt
        return state_next

    def derivative(self, state, action):
        q_init = state[:self.nq]
        v_init = state[self.nq:self.nx]
        pin.computeABADerivatives(
            self.conf.robot.model, self.conf.robot.data,
            np.copy(q_init).astype(np.float32),
            np.copy(v_init).astype(np.float32),
            action.astype(np.float32)
        )
        Fu = np.zeros((self.nx + 1, self.nu))
        Fu[self.nu:-1, :] = self.conf.robot.data.Minv
        Fu[:self.nx, :] *= self.conf.dt
        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1 / self.conf.NORM_ARR[:-1, None])
        return Fu

    def simulate_batch(self, state, action):
        state_next = np.array([self.simulate(s, a) for s, a in zip(state, action)])
        return torch.tensor(state_next, dtype=torch.float32)
    
    def derivative_batch(self, state, action):
        Fu = np.array([self.derivative(s, a) for s, a in zip(state, action)])
        return torch.tensor(Fu, dtype=torch.float32)

    def ee(self, state, recompute=True):
        q = np.array(state[:self.nq])
        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)
        H = self.conf.robot.framePlacement(q.astype(np.float32), RF, recompute)
        return H.translation

    def ee_batch(self, state_batch):
        ee_positions = []
        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)
        for state in state_batch:
            q = np.array(state[:self.nq])
            H = self.conf.robot.framePlacement(q.astype(np.float32), RF)
            ee_positions.append(torch.tensor(H.translation, dtype=torch.float32))
        return torch.stack(ee_positions, dim=0)

    def reward(self, state, action=None):
        # NOTE: QD is velocity cost, we (maybe) should change the notation to dQ for readability.
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = -0.5 * QD_cost * np.sum(state[self.nx // 2:] ** 2)
        total_cost += -0.5 * np.sum((self.ee(state) - self.goal_ee) ** 2)
        if action is not None:
            total_cost += -0.5 * R_cost * np.sum(action ** 2)
        return total_cost

    def reward_batch(self, state_batch, action_batch=None):
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = -0.5 * QD_cost * torch.sum(state_batch[:, self.nx // 2:] ** 2, dim=1)
        ee_pos = self.ee_batch(state_batch) # find the end-effector position for each state in the batch
        target = torch.tensor(self.goal_ee, dtype=ee_pos.dtype, device=ee_pos.device)
        total_cost += -0.5 * torch.sum((ee_pos - target) ** 2, dim=1)
        if action_batch is not None:
            total_cost += -0.5 * R_cost * torch.sum(action_batch ** 2, dim=1)
        return total_cost.unsqueeze(1)