import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import torch
import os
import pinocchio as pin
from confs.base_env import BaseEnv

#----- NN params-----------------------------------------------------------------------------------
NN_LOOPS = np.arange(5000, 500000, 1000)                                                           # Number of updates K of critic and actor performed every TO_EPISODES                                                                              
NN_LOOPS_TOTAL = 500000                                                                            # Max NNs updates total
BATCH_SIZE = 128                                                                                   # Num. of transitions sampled from buffer for each NN update
NH1 = 256                                                                                          # 1st hidden layer size - actor
NH2 = 256                                                                                          # 2nd hidden layer size - actor
NN_PATH = 'iiwa'                                                                                   # Path to save the .pth files for actor and critic
CRITIC_LEARNING_RATE = 5e-4                                                                        # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-4                                                                         # Learning rate for the policy network
bound_NN_action = False                                                                            # Flag to bound the action output by the NN
MAX_NORM_A = 1.0                                                                                   # Maximum norm of gradient for actor
MAX_NORM_C = 1.0                                                                                   # Maximum norm of gradient for critic

#-----IIWA-specific params-------------------------------------------------------------------------
URDF_PATH = os.path.join(os.path.dirname(__file__), 'iiwa.urdf')
robot = RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[os.path.dirname(URDF_PATH)])
robot_data = robot.model.createData()
end_effector_frame_id = 'iiwa_link_7'
q_neutral = pin.neutral(robot.model)
pin.forwardKinematics(robot.model, robot_data, q_neutral)
pin.updateFramePlacements(robot.model, robot_data)
eef_frame_id = robot.model.getFrameId(end_effector_frame_id)
goal_ee = robot_data.oMf[eef_frame_id].translation
nx = 14                                                                                            # Number of state variables (7 joint positions + 7 joint velocities)
nq = 7                                                                                             # Number of joint positions (KUKA IIWA has 7 joints)
nu = 7                                                                                             # Number of actions (controls (torques for each joint))
nv = 7                                                                                             # Number of joint velocities (KUKA IIWA has 7 joints)
X_MIN = np.zeros(nx)
X_MAX = np.zeros(nx)
for joint in robot.model.joints[1:]:
    q_idx = joint.idx_q
    v_idx = joint.idx_v
    X_MIN[q_idx] = robot.model.lowerPositionLimit[q_idx]
    X_MAX[q_idx] = robot.model.upperPositionLimit[q_idx]
    X_MIN[nq + v_idx] = -robot.model.velocityLimit[v_idx]
    X_MAX[nq + v_idx] =  robot.model.velocityLimit[v_idx]

#-----TO params------------------------------------------------------------------------------------
TO_EPISODES = 100                                                                                  # Number of episodes solving TO/computing reward before updating critic and actor
dt = 0.01                                                                                          # timestep
NSTEPS = 32                                                                                        # Max trajectory length
X_INIT_MIN = np.concatenate([X_MIN/4, [0.0]])
X_INIT_MAX = np.concatenate([X_MAX/8, [(NSTEPS-1)*dt]])

#-----Misc params----------------------------------------------------------------------------------
REPLAY_SIZE = 2**16                                                                                # Size of the replay buffer
MC = 0                                                                                             # Flag to use MC or TD(n)
UPDATE_RATE = 0.001                                                                                # Homotopy rate to update the target critic network if TD(n) is used
NSTEPS_TD_N = int(NSTEPS/4)  
NORMALIZE_INPUTS = 0                                                                               # Flag to normalize inputs (state)
NORM_ARR = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10, int(NSTEPS*dt)])                   # Array of values to normalize by

#-----env functions--------------------------------------------------------------------------------
class IiwaEnv(BaseEnv):
    def __init__(self, conf):
        super().__init__(conf)
        self.nx = conf.nx
        self.nq = conf.nq
        self.nu = conf.nu
        self.goal_ee = conf.goal_ee
        self.q_max = conf.X_MAX[:self.nq]
        self.q_min = conf.X_MIN[:self.nq]
        self.v_max = conf.X_MAX[self.nq:self.nx]
        self.v_min = conf.X_MIN[self.nq:self.nx]

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
        q = state[:self.nq]
        v = state[self.nq:self.nx]
    
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().numpy()
        q = np.asarray(q, dtype=np.float64)

        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float64)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float64)

        qdd = pin.aba(self.conf.robot.model, self.conf.robot_data, q, v, action)
        v_new = v + qdd * self.conf.dt
        q_new = pin.integrate(self.conf.robot.model, q, v_new * self.conf.dt)

        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(q_new), np.copy(v_new)
        state_next[-1] = state[-1] + self.conf.dt
        return state_next

    def derivative(self, state, action):
        q_init = state[:self.nq]
        if isinstance(q_init, torch.Tensor):
            q_init = q_init.detach().cpu().numpy()
        v_init = state[self.nq:self.nx]
        if isinstance(v_init, torch.Tensor):
            v_init = v_init.detach().cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        q_init = np.asarray(q_init, dtype=np.float64)
        v_init = np.asarray(v_init, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

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
        QD_cost, R_cost = 0.01, 1e-5
        total_cost = -0.5 * QD_cost * np.sum(state[self.nx // 2:] ** 2)
        total_cost += -0.5 * np.sum((self.ee(state) - self.goal_ee) ** 2)
        if action is not None:
            total_cost += -0.5 * R_cost * np.sum(action ** 2)
        return total_cost

    def reward_batch(self, state_batch, action_batch=None):
        QD_cost, R_cost = 0.01, 1e-5
        total_cost = -0.5 * QD_cost * torch.sum(state_batch[:, self.nx // 2:] ** 2, dim=1)
        ee_pos = self.ee_batch(state_batch) # find the end-effector position for each state in the batch
        target = torch.tensor(self.goal_ee, dtype=ee_pos.dtype, device=ee_pos.device)
        total_cost += -0.5 * torch.sum((ee_pos - target) ** 2, dim=1)
        if action_batch is not None:
            total_cost += -0.5 * R_cost * torch.sum(action_batch ** 2, dim=1)
        return total_cost.unsqueeze(1)