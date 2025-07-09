import numpy as np
import torch
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, inspect, types


class RLTrainer:
    def __init__(self, env, NN, conf, N_try):
        self.env = env
        self.NN = NN
        self.conf = conf
        self.path = f"{self.conf.NN_PATH}/{N_try}"
        os.makedirs(self.path, exist_ok=True)
        self.state_dim = conf.nx + 1
        self.actor_model = None
        self.critic_model = None
        self.target_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        # logs for plots
        self.critic_loss_log = []
        self.actor_loss_log = []
        self.return_log = []
        return
    
    def setup_model(self):
        # Create actor, critic and target NNs
        self.actor_model = self.NN.create_actor()
        self.critic_model = self.NN.create_critic_sine()
        self.target_critic = self.NN.create_critic_sine()

        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.conf.ACTOR_LEARNING_RATE, eps=1e-7)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.conf.CRITIC_LEARNING_RATE, eps=1e-7)
        self.target_critic.load_state_dict(self.critic_model.state_dict())   

    def update(self, states, next_states, partial_rtg, dones, weights):
        '''
        Update both actor and critic networks
        '''
        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss, rtg, values, target_values = self.NN.compute_critic_grad(
            self.critic_model, self.target_critic,
            states, next_states, partial_rtg, dones, weights
        )
        self.critic_optimizer.step()
        self.critic_loss_log.append(critic_loss)

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss = self.NN.compute_actor_grad(self.actor_model, self.critic_model, states)
        self.actor_optimizer.step()
        self.actor_loss_log.append(actor_loss)

        return rtg, values, target_values

    def update_target(self, target_params, source_params):
        '''
        Update target critic network parameters.
        '''
        tau = self.conf.UPDATE_RATE
        with torch.no_grad():
            for t, s in zip(target_params, source_params):
                t.data.copy_(tau * s.data + (1 - tau) * t.data)

    def learn_and_update(self, step_counter, buffer, ep):
        t_sample, t_update, t_target = [], [], []

        for _ in range(int(self.conf.NN_LOOPS[ep])):
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

    def compute_partial_rtg(self, actions, states):
        """
        Computes partial reward-to-go using n-step TD for a given trajectory segment.

        Args:
            actions (np.ndarray): shape (n-1, action_dim)
            states (np.ndarray): shape (n, state_dim)

        Returns:
            states (np.ndarray): Full state rollout (n, state_dim)
            rtg (np.ndarray): Partial reward-to-go estimates (n,)
            next_states (np.ndarray): Bootstrap next states (n, state_dim)
            dones (np.ndarray): Boolean flags for terminal bootstrapping (n,)
            rewards (np.ndarray): Per-step rewards (n + 1,)
        """
        n = self.conf.NSTEPS - int(states[0, -1] / self.conf.dt)
        rewards = np.empty(n)
        next_states = np.zeros((n, self.state_dim))
        dones = np.zeros(n)

        # rollout and get per-step rewards
        for t in range(n-1):
            states[t + 1] = self.env.simulate(states[t], actions[t])
            rewards[t] = self.env.reward(states[t], actions[t])
        rewards[-1] = self.env.reward(states[-1])

        # compute partial reward-to-go
        rtg = np.array([rewards[i:min(i + self.conf.NSTEPS_TD_N, n + 1)].sum()
                        for i in range(n)])
        for i in range(n):
            done = self.conf.MC or i + self.conf.NSTEPS_TD_N >= n
            dones[i] = int(done)
            if not done:
                next_states[i] = states[i + self.conf.NSTEPS_TD_N]
        return states[:-1], rtg, next_states, dones, rewards
    
    def save_weights(self, step='final'):            
        torch.save(self.actor_model.state_dict(), f"{self.path}/actor_{step}.pth")
        torch.save(self.critic_model.state_dict(), f"{self.path}/critic_{step}.pth")
        torch.save(self.target_critic.state_dict(), f"{self.path}/target_critic_{step}.pth")
        print(f"Models saved to {self.path}.")

    def save_conf(self):
        def is_json_serializable(val):
            try:
                json.dumps(val)
                return True
            except (TypeError, OverflowError):
                return False

        conf = {}
        for k, v in vars(self.conf).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (types.FunctionType, types.ModuleType)):
                continue
            if inspect.isclass(v) or inspect.ismethod(v):
                continue
            if isinstance(v, (np.ndarray, torch.Tensor)):
                v = v.tolist()
            if not is_json_serializable(v):
                continue
            conf[k] = v

        save_path = os.path.join(self.path, "conf.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(conf, f, indent=2)
        print(f"Config hyper-parameters saved to {save_path}.")

    def create_TO_init(self, ep, init_state):
        """
        Create initial trajectory for the TrajOpt solver.
        Args:
            ep (int): Current episode number.
            init_state (np.ndarray): Initial state with shape (state_dim,).
        Returns:
            states (np.ndarray): Full state trajectory (n+1, state_dim).
            actions (np.ndarray): Full action trajectory (n, action_dim).
            success (int): 1 if trajectory is valid, 0 otherwise.
        """
        n = self.conf.NSTEPS - int(init_state[-1] / self.conf.dt)
        if n <= 0:
            return None, None, None, 0

        actions = np.zeros((n, self.conf.nu))
        states = np.zeros((n + 1, self.state_dim))
        states[0] = init_state

        for i in range(n):
            if ep == 0:
                actions[i] = np.zeros(self.conf.nu)
            else:
                state_tensor = torch.tensor(states[i][None], dtype=torch.float32)
                actions[i] = self.NN.eval(self.actor_model, state_tensor)\
                                .squeeze().cpu().detach().numpy()
            states[i + 1] = self.env.simulate(states[i], actions[i])

            if np.isnan(states[i + 1]).any():
                return None, None, 0

        return states, actions, 1

    def plot_training_curves(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.critic_loss_log, label="Critic loss")
        plt.plot(self.actor_loss_log,  label="Actor loss")
        plt.xlabel("Gradient update step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.path}/loss_curves.png", dpi=300)
        plt.show()      
        print(f"Training curves saved to {self.path}/loss_curves.png.")   