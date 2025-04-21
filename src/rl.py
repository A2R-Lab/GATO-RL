import uuid
import math
import numpy as np
import torch
import time

class RL_AC:
    def __init__(self, env, NN, conf, N_try):
        self.env = env
        self.NN = NN
        self.conf = conf
        self.N_try = N_try

        self.actor_model = None
        self.critic_model = None
        self.target_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.NSTEPS_SH = 0
        return
    
    def setup_model(self, recover_training=None, weights=None):
        ''' Setup RL model '''
        # Create actor, critic and target NNs
        critic_funcs = {
            'elu': self.NN.create_critic_elu,
            'sine': self.NN.create_critic_sine,
            'sine-elu': self.NN.create_critic_sine_elu,
            'relu': self.NN.create_critic_relu
        }
        if weights is not None:
            self.actor_model = self.NN.create_actor(weights = weights[0])
            self.critic_model = critic_funcs[self.conf.critic_type](weights = weights[1])
            self.target_critic = critic_funcs[self.conf.critic_type](weights = weights[2])
        else:
            self.actor_model = self.NN.create_actor()
            self.critic_model = critic_funcs[self.conf.critic_type]()
            self.target_critic = critic_funcs[self.conf.critic_type]()

        # Initialize optimizers
        self.critic_optimizer   = torch.optim.Adam(self.critic_model.parameters(), eps = 1e-7,\
            lr = self.conf.CRITIC_LEARNING_RATE)
        self.actor_optimizer    = torch.optim.Adam(self.actor_model.parameters(), eps = 1e-7,\
            lr = self.conf.ACTOR_LEARNING_RATE)
        # Set lr schedulers
        if self.conf.LR_SCHEDULE:
            # Piecewise constant decay schedule
            #NOTE: not sure about epochs used in 'milestones' variable
            self.CRITIC_LR_SCHEDULE = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones =\
                 self.conf.values_schedule_LR_C, gamma = 0.5)
            self.ACTOR_LR_SCHEDULE  = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones =\
                self.conf.values_schedule_LR_A, gamma = 0.5)

        # Set initial weights of the NNs
        if recover_training is not None: 
            #NOTE: this was not tested
            NNs_path_rec = str(recover_training[0])
            N_try = recover_training[1]
            update_step_counter = recover_training[2]   
            self.actor_model.load_state_dict(torch.load(f"{NNs_path_rec}/N_try_{N_try}/actor_{update_step_counter}.pth"))
            self.critic_model.load_state_dict(torch.load(f"{NNs_path_rec}/N_try_{N_try}/critic_{update_step_counter}.pth"))
            self.target_critic.load_state_dict(torch.load(f"{NNs_path_rec}/N_try_{N_try}/target_critic_{update_step_counter}.pth"))
        else:
            self.target_critic.load_state_dict(self.critic_model.state_dict())   

    def update(self, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, d_batch, weights_batch,\
        batch_size=None):
        ''' Update both critic and actor '''

        # Update the critic by backpropagating the gradients
        self.critic_optimizer.zero_grad()
        reward_to_go_batch, critic_value, target_critic_value = self.NN.compute_critic_grad(self.critic_model,\
            self.target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, d_batch, weights_batch)
        self.critic_optimizer.step()  # Update the weights
        
        # Update the actor by backpropagating the gradients
        self.actor_optimizer.zero_grad()
        self.NN.compute_actor_grad(self.actor_model, self.critic_model, state_batch, batch_size)

        self.actor_optimizer.step()  # Update the weights
        if self.conf.LR_SCHEDULE:
            self.ACTOR_LR_SCHEDULE.step()
            self.CRITIC_LR_SCHEDULE.step()

        return reward_to_go_batch, critic_value, target_critic_value
        
    def update_target(self, target_weights, weights):
        ''' Update target critic NN '''
        tau = self.conf.UPDATE_RATE
        with torch.no_grad():
            for target_param, param in zip(target_weights, weights):
                target_param.data.copy_(param.data * tau + target_param.data * (1 - tau))

    def learn_and_update(self, update_step_counter, buffer, ep):
        #Tested Successfully# Although only for one iteration (?)
        ''' Sample experience and update buffer priorities and NNs '''
        times_sample = np.zeros(int(self.conf.UPDATE_LOOPS[ep]))
        times_update = np.zeros(int(self.conf.UPDATE_LOOPS[ep]))
        times_update_target = np.zeros(int(self.conf.UPDATE_LOOPS[ep]))
        for i in range(int(self.conf.UPDATE_LOOPS[ep])):
            # Sample batch of transitions from the buffer
            st = time.time()
            state_batch, partial_reward_to_go_batch, state_next_rollout_batch, d_batch, weights_batch, batch_idxes =\
                buffer.sample()
            et = time.time()
            times_sample[i] = et-st
            
            # Update both critic and actor
            st = time.time()
            reward_to_go_batch, critic_value, target_critic_value = self.update(state_batch, state_next_rollout_batch,\
                partial_reward_to_go_batch, d_batch, weights_batch)
            et = time.time()
            times_update[i] = et-st

            # Update target critic
            if not self.conf.MC:
                st = time.time()
                self.update_target(self.target_critic.parameters(), self.critic_model.parameters())
                et = time.time()
                times_update_target[i] = et-st

            update_step_counter += 1

        print(f"Sample times - Avg: {np.mean(times_sample)}; Max:{np.max(times_sample)}; Min: {np.min(times_sample)}\n")
        print(f"Update times - Avg: {np.mean(times_update)}; Max:{np.max(times_update)}; Min: {np.min(times_update)}\n")
        print(f"Target Update times - Avg: {np.mean(times_update_target)}; Max:{np.max(times_update_target)}; Min: {np.min(times_update_target)}\n")
        return update_step_counter
    
    def RL_Solve(self, TO_controls, TO_states):
        NSTEPS_SH = self.conf.NSTEPS - int(TO_states[0, -1] / self.conf.dt)
        rwrd_arr = np.empty(NSTEPS_SH + 1)
        next_arr = np.zeros((NSTEPS_SH + 1, self.conf.nb_state))
        go_arr = np.empty(NSTEPS_SH + 1)
        done_arr = np.zeros(NSTEPS_SH + 1)
        ee_arr = np.empty((NSTEPS_SH + 1, 3))

        # start RL episode
        for t in range(NSTEPS_SH):
            u = TO_controls[t if t < NSTEPS_SH - 1 else t - 1]
            TO_states[t + 1], rwrd_arr[t] = self.env.step(TO_states[t], u)
            ee_arr[t + 1] = self.env.ee(TO_states[t + 1])
        rwrd_arr[-1] = self.env.reward(TO_states[-1])

        # compute partial cost-to-go (n-step TD or monte carlo)
        for i in range(NSTEPS_SH + 1):
            final = NSTEPS_SH if self.conf.MC else min(i + self.conf.nsteps_TD_N, NSTEPS_SH)
            done_arr[i] = int(self.conf.MC or final == NSTEPS_SH)
            if not self.conf.MC and final < NSTEPS_SH:
                next_arr[i] = TO_states[final + 1]
            go_arr[i] = rwrd_arr[i:final + 1].sum()

        return TO_states, go_arr, next_arr, done_arr, rwrd_arr, ee_arr
    
    def RL_save_weights(self, update_step_counter='final'):
        ''' Save NN weights '''
        actor_model_path = f"{self.conf.NNs_path}/N_try_{self.N_try}/actor_{update_step_counter}.pth"
        critic_model_path = f"{self.conf.NNs_path}/N_try_{self.N_try}/critic_{update_step_counter}.pth"
        target_critic_path = f"{self.conf.NNs_path}/N_try_{self.N_try}/target_critic_{update_step_counter}.pth"

        # Save model weights
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.critic_model.state_dict(), critic_model_path)
        torch.save(self.target_critic.state_dict(), target_critic_path)

    def create_TO_init(self, ep, ICS):
        ''' Create initial state and initial controls for TO '''
        init_rand_state = ICS    
        NSTEPS_SH = self.conf.NSTEPS - int(init_rand_state[-1]/self.conf.dt)
        if NSTEPS_SH == 0:
            return None, None, None, 0

        # Initialize array to initialize TO state and control variables
        init_TO_controls = np.zeros((NSTEPS_SH, self.conf.nb_action))
        init_TO_states = np.zeros(( NSTEPS_SH+1, self.conf.nb_state)) 
        init_TO_states[0,:] = init_rand_state

        # Simulate actor's actions to compute trajectory used to initialize TO state variables
        for i in range(NSTEPS_SH):   
            init_TO_controls[i,:] = np.zeros(self.conf.nb_action) if ep == 0 else\
                self.NN.eval(self.actor_model, torch.tensor(np.array([init_TO_states[i,:]]),\
                dtype=torch.float32)).squeeze().detach().cpu().numpy()
            print(f"init TO controls {i+1}/{NSTEPS_SH}:  {init_TO_controls[i,:]}")
            init_TO_states[i+1,:] = self.env.simulate(init_TO_states[i,:],init_TO_controls[i,:])

            if np.isnan(init_TO_states[i+1,:]).any():
                return None, None, None, 0

        return init_rand_state, init_TO_states, init_TO_controls, 1