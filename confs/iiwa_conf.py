import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import torch
import os
import pinocchio as pin

#-----CACTO params---------------------------------------------------------------------------------
EP_UPDATE = 10                                                                                     # Number of episodes before updating critic and actor
NUPDATES = 100000                                                                                  # Max NNs updates
UPDATE_LOOPS = np.arange(1000, 48000, 3000)                                                        # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                                
NEPISODES = int(EP_UPDATE*len(UPDATE_LOOPS))                                                       # Max training episodes
NLOOPS = len(UPDATE_LOOPS)                                                                         # Number of algorithm loops
NSTEPS = 50                                                                                        # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                                        # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                         # Learning rate for the policy network
REPLAY_SIZE = 2**16                                                                                # Size of the replay buffer
BATCH_SIZE = 128
MC = 0                                                                                             # Flag to use MC or TD(n)
UPDATE_RATE = 0.001                                                                                # Homotopy rate to update the target critic network if TD(n) is used
nsteps_TD_N = int(NSTEPS/4)  

#----- NN params-----------------------------------------------------------------------------------
critic_type = 'sine'                                                                               # Activation function - critic (either relu, elu, sine, sine-elu)
NH1 = 256                                                                                          # 1st hidden layer size - actor
NH2 = 256                                                                                          # 2nd hidden layer size - actor
NNs_path = 'iiwa'
NORMALIZE_INPUTS = 0                                                                               # Flag to normalize inputs (state)
kreg_l1_A = 1e-2                                                                                   # Weight of L1 regularization in actor's network - kernel
kreg_l2_A = 1e-2                                                                                   # Weight of L2 regularization in actor's network - kernel
breg_l1_A = 1e-2                                                                                   # Weight of L2 regularization in actor's network - bias
breg_l2_A = 1e-2                                                                                   # Weight of L2 regularization in actor's network - bias
kreg_l1_C = 1e-2                                                                                   # Weight of L1 regularization in critic's network - kernel
kreg_l2_C = 1e-2                                                                                   # Weight of L2 regularization in critic's network - kernel
breg_l1_C = 1e-2                                                                                   # Weight of L1 regularization in critic's network - bias
breg_l2_C = 1e-2                                                                                   # Weight of L2 regularization in critic's network - bias

#-----TO params------------------------------------------------------------------------------------
dt = 0.01
x_min = np.array([-2.967,-2.094,-2.967,-2.094,-2.967,-2.094,-3.054,
                    -1.57,-1.57,-1.57,-1.57,-1.57,-1.57,-1.57,0])
x_init_min = np.array([-2.967,-2.094,-2.967,-2.094,-2.967,-2.094,-3.054,
                    1.57,1.57,1.57,1.57,1.57,1.57,1.57,0])
x_max = np.array([2.967,2.094,2.967,2.094,2.967,2.094,3.054,
                    1.57,1.57,1.57,1.57,1.57,1.57,1.57,np.inf])
x_init_max = np.array([2.967,2.094,2.967,2.094,2.967,2.094,3.054,
                    1.57,1.57,1.57,1.57,1.57,1.57,1.57,(NSTEPS-1)*dt])
state_norm_arr = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10, int(NSTEPS*dt)])
goal_ee = [0.5, 0.5, 0.5]
state_dim = 15
nx = 14
nq = 7
na = 7
URDF_PATH = os.path.abspath(os.path.join('confs', 'iiwa.urdf'))
robot = RobotWrapper.BuildFromURDF(URDF_PATH, package_dirs=[os.path.dirname(URDF_PATH)])
robot_data = robot.model.createData()
end_effector_frame_id = 'iiwa_link_7'

#-----env functions--------------------------------------------------------------------------------
class Env:
    """
    This class provides functions related to the dynamics and reward computations, mainly used for
    computing the reward-to-go (rl_trainer.compute_partial_rtg()) and the actor network update
    (neural_network.compute_actor_grad()). Also, to create random initial states at each episode
    for TO to solve.
    """
    def __init__(self, conf):
        self.conf = conf
        self.nq = conf.nq # Number of joints in the robot (7 for iiwa)
        self.nx = conf.nx # Number of state variables (14 for iiwa: 7 joint positions + 7 velocities)
        self.na = conf.na # Number of actuators/control inputs (equal to nq since 7 actuated joints)
        self.TARGET_STATE = conf.goal_ee

    def reset_batch(self, batch_size):
        """
        Reset the environment to a random initial state for a batch of size `batch_size`.
        This generates 10 different intial states for the robot, each with a random time value.

        TODO: Need to ask Seyoung why do we initialize a random time value here.
        Args:
            batch_size (int): Number of initial states to reset

        Returns:
            np.ndarray: Array of shape (batch_size, nx+1), where each row is the state
                        appended by the timestep
        """
        # Generate random time values for each episode in the batch
        # These represent how much time has elapsed since episode start (0 to max episode time)
        times = np.random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1], batch_size)
        
        # Generate random robot configurations (joint positions + velocities) for each episode
        # Shape: (batch_size, 14) where 14 = 7 joint positions + 7 joint velocities
        states = np.random.uniform(self.conf.x_init_min[:-1], self.conf.x_init_max[:-1],
                                    size=(batch_size, len(self.conf.x_init_max[:-1])))
        
        # Quantize the continuous time values to discrete timesteps aligned with dt
        # This ensures all time values are multiples of dt (e.g., 0.00, 0.01, 0.02, etc.)
        times_int = np.expand_dims(self.conf.dt*np.round(times/self.conf.dt), axis=1)
        
        # Combine robot states with their corresponding time values
        # Final shape: (batch_size, 15) where 15 = 14 state variables + 1 time variable
        return np.hstack((states, times_int))

    def simulate(self, state, action):
        """
        Simulates one timestep forward given state and action.

        Args:
            state (np.ndarray): Current state (size nx+1)
            action (np.ndarray): Control input (size na)

        Returns:
            np.ndarray: Next state (size nx+1)
        """
        state_next = np.zeros(self.nx+1)
        q, v = state[:self.nq], state[self.nq:self.nx]
        qdd = pin.aba(self.conf.robot.model, self.conf.robot_data, q, v, action)
        v_new = v + qdd * self.conf.dt
        q_new = pin.integrate(self.conf.robot.model, q, v_new * self.conf.dt)

        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(q_new), np.copy(v_new)
        state_next[-1] = state[-1] + self.conf.dt
        return state_next
    
    def simulate_batch(self, state, action):
        """
        Batch version of simulate(), used in neural_network.compute_actor_grad().

        Args:
            state (np.ndarray): Batch of states with shape (batch_size, nx+1)
            action (np.ndarray): Batch of actions with shape (batch_size, na)

        Returns:
            torch.Tensor: Batch of next states with shape (batch_size, nx+1)
        """
        state_next = np.array([self.simulate(s, a) for s, a in zip(state, action)])
        return torch.tensor(state_next, dtype=torch.float32)

    def step(self, state, action):
        """
        Takes a step in the environment and computes the immediate reward upon
        taking the action.

        Args:
            state (np.ndarray): Current state of shape (nx+1)
            action (np.ndarray): Control input of shape (na)

        Returns:
            tuple: (next_state, reward) where
                next_state (np.ndarray): Simulated next state.
                reward (float): Immediate reward after taking the action.
        """
        state_next = self.simulate(state, action)
        reward = self.reward( state, action)
        return (state_next, reward)
    
    def derivative(self, state, action):
        """
        Computes the Jacobian of the dynamics w.r.t the control input (ds/da).

        Args:
            state (np.ndarray): Current state
            action (np.ndarray): Control input

        Returns:
            np.ndarray: Derivative matrix (nx+1, na)
        """
        # Create robot model in Pinocchio with q_init as initial configuration
        q_init = state[:self.nq]
        v_init = state[self.nq:self.nx]

        # Dynamics gradient w.r.t control (1st order euler)
        pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data,\
            np.copy(q_init).astype(np.float32), np.copy(v_init).astype(np.float32),
            action.astype(np.float32))       

        Fu = np.zeros((self.nx+1, self.na))
        Fu[self.na:-1, :] = self.conf.robot.data.Minv
        Fu[:self.nx, :] *= self.conf.dt

        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/self.conf.state_norm_arr[:-1,None])  
        return Fu
    
    def derivative_batch(self, state, action):
        """
        Batch version of derivative(), used in neural_network.compute_actor_grad()

        Args:
            state (np.ndarray): Batch of states (batch_size, nx+1)
            action (np.ndarray): Batch of actions (batch_size, na)

        Returns:
            torch.Tensor: Batch of derivative matrices (batch_size, nx+1, na)
        """
        Fu = np.array([self.derivative(s, a) for s, a in zip(state, action)])
        return torch.tensor(Fu, dtype=torch.float32)

    def ee(self, state, recompute=True):
        """
        Computes end-effector position.

        Args:
            state (np.ndarray): Robot state
            recompute (bool): Whether to recompute the kinematics

        Returns:
            np.ndarray: End-effector position
        """
        q = np.array(state[:self.nq])
        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)
        H = self.conf.robot.framePlacement(q.astype(np.float32), RF, recompute)
        return H.translation

    def reward(self, state, action=None):
        """
        Computes reward from state and action, used in compute_partial_rtg().

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray, optional): Control input.

        Returns:
            float: Reward value.
        """
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = -0.5 * QD_cost * np.sum(state[self.nx//2:] ** 2) # v cost
        total_cost += -0.5 * np.sum((self.ee(state) - self.TARGET_STATE) ** 2) # ee pos cost

        if action is not None:
            total_cost += -0.5 * R_cost * np.sum(action ** 2) # control cost

        return total_cost
    
    def reward_batch(self, state_batch, action_batch=None):
        """
        Computes batch of rewards using tensors, used in neural_network.compute_actor_grad()

        Args:
            state_batch (torch.Tensor): Batch of states (batch_size, nx+1)
            action_batch (torch.Tensor, optional): Batch of actions (batch_size, na)

        Returns:
            torch.Tensor: Batch of reward values (batch_size,)
        """
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = -0.5 * QD_cost * torch.sum(state_batch[:, self.nx//2:]**2, dim=1) # v cost

        ee_pos = self.ee_batch(state_batch)
        target = torch.tensor(self.TARGET_STATE, dtype=ee_pos.dtype, device=ee_pos.device)
        total_cost += -0.5 * torch.sum((ee_pos - target) ** 2, dim=1) # ee pos cost

        if action_batch is not None:
            total_cost += -0.5 * R_cost * torch.sum(action_batch ** 2, dim=1)

        return total_cost.unsqueeze(1)

    def cost(self, state, action):
        """
        Computes total cost (non-negative) used for optimization.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Control input.

        Returns:
            float: Total cost.
        """
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = 0.5 * QD_cost * np.sum(state[self.nx//2:] ** 2) # velocity cost
        total_cost += 0.5 * R_cost * np.sum(action ** 2) # v`` control cost
        total_cost += 0.5 * np.sum((self.ee(state) - self.conf.TARGET_STATE) ** 2) # ee pos cost
        return total_cost
