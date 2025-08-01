import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import torch
import os
import pinocchio as pin
from confs.base_env import BaseEnv

#----- NN params-----------------------------------------------------------------------------------
NN_LOOPS = np.arange(1000, 500000, 100)                                                            
NN_LOOPS_TOTAL = 500000
BATCH_SIZE = 128
NH1 = 128
NH2 = 128
NN_PATH = 'planar3dof'
CRITIC_LEARNING_RATE = 1e-4
ACTOR_LEARNING_RATE = 1e-4
bound_NN_action = False
MAX_NORM_A = 1.0
MAX_NORM_C = 1.0

#-----Planar 3DOF-specific params------------------------------------------------------------------
URDF_PATH = os.path.join(os.path.dirname(__file__), 'planar3dof.urdf')
robot = RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[os.path.dirname(URDF_PATH)])
robot_data = robot.model.createData()
q = np.zeros(3)
pin.forwardKinematics(robot.model, robot_data, q)
pin.updateFramePlacements(robot.model, robot_data)
end_effector_frame_id = 'EE'
eef_frame_id = robot.model.getFrameId(end_effector_frame_id)
goal_ee = robot_data.oMf[eef_frame_id].translation
nx = 6
nq = 3
nu = 3
nv = 3

#-----TO params------------------------------------------------------------------------------------
TO_EPISODES = 100
dt = 0.05
NSTEPS = 100
X_MIN = np.array([-np.pi, -np.pi, -np.pi, -20.0, -20.0, -20.0])
X_MAX = np.array([np.pi, np.pi, np.pi, 20.0, 20.0, 20.0])
X_INIT_MIN = np.concatenate([X_MIN/2, [0.0]])
X_INIT_MAX = np.concatenate([X_MAX/2, [(NSTEPS-1)*dt]])

#-----Misc params----------------------------------------------------------------------------------
REPLAY_SIZE = 2**12
MC = 0
UPDATE_RATE = 0.001
NSTEPS_TD_N = int(NSTEPS/4)
NORMALIZE_INPUTS = 0
NORM_ARR = np.array([10,10,10,10,10,10,int(NSTEPS*dt)])

#-----env functions--------------------------------------------------------------------------------
class Planar3DOFEnv(BaseEnv):
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

        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        qdd = pin.aba(self.conf.robot.model, self.conf.robot_data, q, v, action)
        v_new = v + qdd * self.conf.dt
        q_new = pin.integrate(self.conf.robot.model, q, v_new * self.conf.dt)

        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(q_new), np.copy(v_new)
        state_next[-1] = state[-1] + self.conf.dt
        print(state_next)
        return state_next

    def derivative(self, state, action):
        q_init = np.asarray(state[:self.nq], dtype=np.float64)
        v_init = np.asarray(state[self.nq:self.nx], dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        pin.computeABADerivatives(
            self.conf.robot.model, self.conf.robot.data,
            q_init.astype(np.float32),
            v_init.astype(np.float32),
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