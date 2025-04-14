import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import torch
import os
import pinocchio as pin
# import pinocchio.casadi as cpin
import casadi as ca

############################################# CACTO PARAMETERS #############################################
EP_UPDATE = 200                                                                                            # Number of episodes before updating critic and actor
NUPDATES = 100000                                                                                           # Max NNs updates
UPDATE_LOOPS = np.arange(1000, 48000, 3000)                                                                 # Number of updates of both critic and actor performed every EP_UPDATE episodes                                                                                
NEPISODES = int(EP_UPDATE*len(UPDATE_LOOPS))                                                                # Max training episodes
NLOOPS = len(UPDATE_LOOPS)                                                                                  # Number of algorithm loops
NSTEPS = 32                                                                                                 # Max episode length
CRITIC_LEARNING_RATE = 5e-4                                                                                 # Learning rate for the critic network
ACTOR_LEARNING_RATE = 1e-3                                                                                  # Learning rate for the policy network
REPLAY_SIZE = 2**16                                                                                         # Size of the replay buffer
BATCH_SIZE = 128      

# Set _steps_TD_N ONLY if MC not used
MC = 0                                                                                                      # Flag to use MC or TD(n)
if not MC:
    UPDATE_RATE = 0.001                                                                                     # Homotopy rate to update the target critic network if TD(n) is used
    nsteps_TD_N = int(NSTEPS/4)  

############################################# NN PARAMETERS #################################################
critic_type = 'sine'                                                                                        # Activation function - critic (either relu, elu, sine, sine-elu)
NH1 = 256                                                                                                   # 1st hidden layer size - actor
NH2 = 256                                                                                                   # 2nd hidden layer size - actor
                                                                                                  			# 2nd hidden layer size  
LR_SCHEDULE = 0                                                                                             # Flag to use a scheduler for the learning rates
boundaries_schedule_LR_C = [200*REPLAY_SIZE/BATCH_SIZE, 
                            300*REPLAY_SIZE/BATCH_SIZE,
                            400*REPLAY_SIZE/BATCH_SIZE,
                            500*REPLAY_SIZE/BATCH_SIZE]     
# Values of critic LR                            
values_schedule_LR_C = [CRITIC_LEARNING_RATE,
                        CRITIC_LEARNING_RATE/2,
                        CRITIC_LEARNING_RATE/4,
                        CRITIC_LEARNING_RATE/8,
                        CRITIC_LEARNING_RATE/16]  
# Numbers of critic updates after which the actor LR is changed (based on values_schedule_LR_A)
boundaries_schedule_LR_A = [200*REPLAY_SIZE/BATCH_SIZE,
                            300*REPLAY_SIZE/BATCH_SIZE,
                            400*REPLAY_SIZE/BATCH_SIZE,
                            500*REPLAY_SIZE/BATCH_SIZE]   
# Values of actor LR                            
values_schedule_LR_A = [ACTOR_LEARNING_RATE,
                        ACTOR_LEARNING_RATE/2,
                        ACTOR_LEARNING_RATE/4,
                        ACTOR_LEARNING_RATE/8,
                        ACTOR_LEARNING_RATE/16]  

NORMALIZE_INPUTS = 1                                                                                        # Flag to normalize inputs (state)

kreg_l1_A = 1e-2                                                                                            # Weight of L1 regularization in actor's network - kernel
kreg_l2_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - kernel
breg_l1_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - bias
breg_l2_A = 1e-2                                                                                            # Weight of L2 regularization in actor's network - bias
kreg_l1_C = 1e-2                                                                                            # Weight of L1 regularization in critic's network - kernel
kreg_l2_C = 1e-2                                                                                            # Weight of L2 regularization in critic's network - kernel
breg_l1_C = 1e-2                                                                                            # Weight of L1 regularization in critic's network - bias
breg_l2_C = 1e-2                                                                                            # Weight of L2 regularization in critic's network - bias

############################################# BUFFER PARAMETERS #############################################
prioritized_replay_alpha = 0                                                                                # α determines how much prioritization is used, set to 0 to use a normal buffer. Used to define the probability of sampling transition i --> P(i) = p_i**α / sum(p_k**α) where p_i is the priority of transition i 
prioritized_replay_beta = 0.6          
prioritized_replay_beta_iters = None                                                                        # Therefore let's exploit the flexibility of annealing the amount of IS correction over time, by defining a schedule on the exponent β that from its initial value β0 reaches 1 only at the end of learning.
prioritized_replay_eps = 1e-2                                                                               # It's a small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
fresh_factor = 0.95                                                                                         # Refresh factor

############################################# ROBOT PARAMETERS ##############################################
dt = 0.1
nb_state = 15
x_min = np.array([-2.967,-2.094,-2.967,-2.094,-2.967,-2.094,-3.054,-1.57,-1.57,-1.57,-1.57,-1.57,-1.57,-1.57,0])
x_init_min = np.array([-2.967,-2.094,-2.967,-2.094,-2.967,-2.094,-3.054,1.57,1.57,1.57,1.57,1.57,1.57,1.57,0])
x_max = np.array([2.967,2.094,2.967,2.094,2.967,2.094,3.054,1.57,1.57,1.57,1.57,1.57,1.57,1.57,np.inf])
x_init_max = np.array([2.967,2.094,2.967,2.094,2.967,2.094,3.054,1.57,1.57,1.57,1.57,1.57,1.57,1.57,(NSTEPS-1)*dt])
state_norm_arr = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, int(NSTEPS*dt)])
nb_action = 7
nq = 7
nv = 7
nx = 14
na = 7
URDF_PATH = os.path.join(os.getcwd(), 'urdfs/iiwa.urdf')
robot = RobotWrapper.BuildFromURDF(URDF_PATH, [URDF_PATH])
robot_data = robot.model.createData()
end_effector_frame_id = 'iiwa_link_7'
TARGET_STATE = [0.5, 0.5, 0.5]
#############################################################################################################

class Env:
    def __init__(self, conf):
        self.conf = conf
        self.nq = conf.nq
        self.nv = conf.nv
        self.nx = conf.nx
        self.nu = conf.na
        self.TARGET_STATE = self.conf.TARGET_STATE
        # self.cmodel = cpin.Model(self.conf.robot.model)
        # self.cdata = self.cmodel.createData()
        cx = ca.SX.sym("x",self.conf.nx,1)
        cu = ca.SX.sym("u",self.conf.na,1)
        # self.x_next = ca.Function('x_next', [cx, cu], [self.cpin_simulate(cx,cu)])
        # self.ee_p = ca.Function('ee_p', [cx], [self.cdata.oMf[self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)].translation])
        # self.cost = ca.Function('cost', [cx,cu], [self.cost(cx,cu)])

    def reset(self):
        ''' Choose initial state uniformly at random '''
        state = np.zeros(self.conf.nb_state)
        time = np.random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1])
        for i in range(self.conf.nb_state-1): 
            state[i] = np.random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i])
        state[-1] = self.conf.dt*round(time/self.conf.dt)
        return state
    

    def reset_batch(self, batch_size):
        ''' Create batch of random initial states '''
        times = np.random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1], batch_size)
        states = np.random.uniform(self.conf.x_init_min[:-1], self.conf.x_init_max[:-1], size=(batch_size, len(self.conf.x_init_max[:-1])))
        times_int = np.expand_dims(self.conf.dt*np.round(times/self.conf.dt), axis=1)
        return np.hstack((states, times_int))


    def simulate(self, state, action):
        state_next = np.zeros(self.nx+1)
        q, v = state[:self.nq], state[self.nq:self.nx]
        qdd = pin.aba(self.conf.robot.model, self.conf.robot_data, q, v, action)
        v_new = v + qdd * self.conf.dt
        q_new = pin.integrate(self.conf.robot.model, q, v_new * dt)

        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(q_new), np.copy(v_new)
        state_next[-1] = state[-1] + self.conf.dt
        return state_next
    
    def simulate_batch(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        state_next = np.array([self.simulate(s, a) for s, a in zip(state, action)])

        return torch.tensor(state_next, dtype=torch.float32)

    def step(self, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)

        # compute reward
        reward = self.reward( state, action)

        return (state_next, reward)
    
    def derivative(self, state, action):
        ''' Compute the derivative '''
        # Create robot model in Pinocchio with q_init as initial configuration
        q_init = state[:self.nq]
        v_init = state[self.nq:self.nx]

        # Dynamics gradient w.r.t control (1st order euler)
        pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, np.copy(q_init).astype(np.float32), np.copy(v_init).astype(np.float32), action.astype(np.float32))       

        Fu = np.zeros((self.nx+1, self.nu))
        Fu[self.nv:-1, :] = self.conf.robot.data.Minv
        Fu[:self.nx, :] *= self.conf.dt

        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/self.conf.state_norm_arr[:-1,None])  

        return Fu
    
    def derivative_batch(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        Fu = np.array([self.derivative(s, a) for s, a in zip(state, action)])

        return torch.tensor(Fu, dtype=torch.float32)

    def ee(self, state, recompute=True):
        ''' Compute end-effector position '''
        q = np.array(state[:self.nq])
        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)
        H = self.conf.robot.framePlacement(q.astype(np.float32), RF, recompute)
        return H.translation
    
    def ee_batch(self, state_batch):
        ''' Compute end-effector positions for a batch of states (no gradients) '''
        ee_positions = []
        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id)

        for state in state_batch:
            q = np.array(state[:self.nq])
            H = self.conf.robot.framePlacement(q.astype(np.float32), RF)
            ee_positions.append(torch.tensor(H.translation, dtype=torch.float32))

        return torch.stack(ee_positions, dim=0)

    def reward(self, state, action=None):
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = -0.5 * QD_cost * np.sum(state[self.nx//2:] ** 2) # velocity cost
        total_cost += -0.5 * np.sum((self.ee(state) - self.conf.TARGET_STATE) ** 2) # ee pos cost

        if action is not None:
            total_cost += -0.5 * R_cost * np.sum(action ** 2) # control cost

        return total_cost
    
    def reward_batch(self, state_batch, action_batch=None):
        ''' Compute reward using tensors. Batch-wise computation '''
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = -0.5 * QD_cost * torch.sum(state_batch[:, self.nx//2:]**2, dim=1) # velocity cost

        ee_pos = self.ee_batch(state_batch)
        target = torch.tensor(self.conf.TARGET_STATE, dtype=ee_pos.dtype, device=ee_pos.device)
        total_cost += -0.5 * torch.sum((ee_pos - target) ** 2, dim=1) # ee pos cost

        if action_batch is not None:
            total_cost += -0.5 * R_cost * torch.sum(action_batch ** 2, dim=1)

        return total_cost.unsqueeze(1)

    def cost(self, state, action):
        QD_cost, R_cost = 0.0001, 0.0001
        total_cost = 0.5 * QD_cost * np.sum(state[self.nx//2:] ** 2) # velocity cost
        total_cost += 0.5 * R_cost * np.sum(action ** 2) # v`` control cost
        total_cost += 0.5 * np.sum((self.ee(state) - self.conf.TARGET_STATE) ** 2) # ee pos cost
        return total_cost
