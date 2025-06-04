import uuid
import math
import numpy as np
import torch
import time

class RLTrainer:
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
    
    def setup_model(self, recover_training=None):
        # Create actor, critic and target NNs
        self.actor_model = self.NN.create_actor()
        self.critic_model = self.NN.create_critic_sine()
        self.target_critic = self.NN.create_critic_sine()

        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.conf.ACTOR_LEARNING_RATE, eps=1e-7)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.conf.CRITIC_LEARNING_RATE, eps=1e-7)

        #  Recover weights
        if recover_training is not None:
            path, N_try, step = recover_training
            self.actor_model.load_state_dict(torch.load(f"{path}/N_try_{N_try}/actor_{step}.pth"))
            self.critic_model.load_state_dict(torch.load(f"{path}/N_try_{N_try}/critic_{step}.pth"))
            self.target_critic.load_state_dict(torch.load(f"{path}/N_try_{N_try}/target_critic_{step}.pth"))
        else:
            self.target_critic.load_state_dict(self.critic_model.state_dict())   

    def update(self, states, next_states, partial_rtg, dones, weights):
        # update critic
        self.critic_optimizer.zero_grad()
        rtg, values, target_values = self.NN.compute_critic_grad(
            self.critic_model, self.target_critic,
            states, next_states, partial_rtg, dones, weights
        )
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer.zero_grad()
        self.NN.compute_actor_grad(self.actor_model, self.critic_model, states)
        self.actor_optimizer.step()

        return rtg, values, target_values

    def update_target(self, target_params, source_params):
        tau = self.conf.UPDATE_RATE
        with torch.no_grad():
            for t, s in zip(target_params, source_params):
                t.data.copy_(tau * s.data + (1 - tau) * t.data)

    def learn_and_update(self, step_counter, buffer, ep):
        t_sample, t_update, t_target = [], [], []

        for _ in range(int(self.conf.UPDATE_LOOPS[ep])):
            # sample from buffer
            t0 = time.time()
            s, prtg, s_next, d, w = buffer.sample()
            t_sample.append(time.time() - t0)

            # update critic and actor
            t1 = time.time()
            rtg, values, target_values = self.update(s, s_next, prtg, d, w)
            t_update.append(time.time() - t1)

            # update target critic
            if not self.conf.MC:
                t2 = time.time()
                self.update_target(self.target_critic.parameters(), self.critic_model.parameters())
                t_target.append(time.time() - t2)

            step_counter += 1

        print(f"Sample time avg: {np.mean(t_sample):.4f}s")
        print(f"Update time avg: {np.mean(t_update):.4f}s")
        print(f"Target update time avg: {np.mean(t_target):.4f}s")
        return step_counter

    def RL_Solve(self, actions, states):
        n = self.conf.NSTEPS - int(states[0, -1] / self.conf.dt)
        rewards = np.empty(n + 1)
        next_states = np.zeros((n + 1, self.conf.state_dim))
        dones = np.zeros(n + 1)

        # start RL episode
        for t in range(n):
            u = actions[t if t < n - 1 else t - 1]
            states[t + 1], rewards[t] = self.env.step(states[t], u)
        rewards[-1] = self.env.reward(states[-1])

        # compute partial cost-to-go
        rtg = np.array([rewards[i:min(i + self.conf.nsteps_TD_N, n)].sum()
                        for i in range(n + 1)])
        for i in range(n + 1):
            done = self.conf.MC or i + self.conf.nsteps_TD_N >= n
            dones[i] = int(done)
            if not done:
                next_states[i] = states[i + self.conf.nsteps_TD_N]

        return states, rtg, next_states, dones, rewards
    
    def RL_save_weights(self, step='final'):
            path = f"{self.conf.NNs_path}/N_try_{self.N_try}"
            torch.save(self.actor_model.state_dict(), f"{path}/actor_{step}.pth")
            torch.save(self.critic_model.state_dict(), f"{path}/critic_{step}.pth")
            torch.save(self.target_critic.state_dict(), f"{path}/target_critic_{step}.pth")

    def create_TO_init(self, ep, init_state):
        n = self.conf.NSTEPS - int(init_state[-1] / self.conf.dt)
        if n <= 0:
            return None, None, None, 0

        actions = np.zeros((n, self.conf.na))
        states = np.zeros((n + 1, self.conf.state_dim))
        states[0] = init_state

        for i in range(n):
            if ep == 0:
                actions[i] = np.zeros(self.conf.na)
            else:
                state_tensor = torch.tensor(states[i][None], dtype=torch.float32)
                actions[i] = self.NN.eval(self.actor_model, state_tensor)\
                                .squeeze().cpu().detach().numpy()
            print(f"init TO controls {i + 1}/{n}: {actions[i]}")
            states[i + 1] = self.env.simulate(states[i], actions[i])

            if np.isnan(states[i + 1]).any():
                return None, None, None, 0

        return init_state, states, actions, 1
