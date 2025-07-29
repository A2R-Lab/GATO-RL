import numpy as np
import torch
import torch.nn as nn
from siren_pytorch import Siren
from utils import normalize_tensor


class ActorCriticNet:
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.MSE = nn.MSELoss()
        self.batch_size = conf.BATCH_SIZE
        self.state_dim = conf.nx + 1

    def create_actor(self, device='cpu', dtype=torch.float32):
        model = nn.Sequential(
            nn.Linear(self.state_dim, self.conf.NH1),
            nn.LayerNorm(self.conf.NH1),
            nn.ELU(inplace=True),

            nn.Linear(self.conf.NH1, self.conf.NH2),
            nn.LayerNorm(self.conf.NH2),
            nn.ELU(inplace=True),

            nn.Linear(self.conf.NH2, self.conf.nu * self.conf.NSTEPS)
        )

        # Weight initialization
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

        return model.to(device=device, dtype=dtype)

    def create_critic(self, device='cpu', dtype=torch.float32):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(inplace=True),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ELU(inplace=True),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ELU(inplace=True),
            
            nn.Linear(128, 1)
        )

        # Weight initialization
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

        return model.to(device=device, dtype=dtype)

    def normalize_tensor(state, state_norm_arr):
        state_norm_time = torch.cat([
            torch.zeros([state.shape[0], state.shape[1] - 1]),
            torch.reshape((state[:, -1] / state_norm_arr[-1]) * 2 - 1, (state.shape[0], 1))
        ], dim=1)
        
        state_norm_no_time = state / state_norm_arr
        mask = torch.cat([
            torch.ones([state.shape[0], state.shape[1] - 1]),
            torch.zeros([state.shape[0], 1])
        ], dim=1)
        
        state_norm = state_norm_no_time * mask + state_norm_time * (1 - mask)
        return state_norm.to(torch.float32)

    def eval(self, NN, input, is_actor=False):
        if not torch.is_tensor(input):
            input = torch.tensor(np.array(input), dtype=torch.float32)

        if self.conf.NORMALIZE_INPUTS:
            norm_arr = torch.tensor(self.conf.NORM_ARR, dtype=torch.float32)
            input = normalize_tensor(input, norm_arr)
        output = NN(input)

        # Enforce action bounds
        if is_actor and getattr(self.conf, 'bound_NN_action', False):
            output = torch.tanh(output) * self.conf.u_max

        if is_actor:
            B = input.shape[0]
            H = self.conf.NSTEPS
            nu = self.conf.nu
            output = output.view(B, H, nu)

        return output

    def compute_critic_grad(self, critic, target_critic, states, next_states,
                            partial_rtg, dones, weights):
        if self.conf.MC:
            full_rtg = partial_rtg
        else:
            # if n-step TD, add remaining reward-to-go using target critic
            next_vals = self.eval(target_critic, next_states).detach()
            full_rtg = partial_rtg + (1 - dones) * next_vals

        values = self.eval(critic, states)
        loss = self.MSE(values, full_rtg)
        critic.zero_grad()
        loss.backward()

        return loss.item(), full_rtg, values, self.eval(target_critic, states)

    def compute_actor_grad(self, actor, critic, states):
        """
        Computes the actor gradient using n-step returns with separate actor horizon H:
            L = - (r_0 + γ r_1 + ... + γ^{n-1} r_{n-1} + γ^n V(s_n))

        Args:
            actor: The actor network predicting (H, action_dim) per state.
            critic: The critic network used to estimate V(s_n).
            states (torch.Tensor): shape (B, state_dim)

        Returns:
            loss.item(): scalar loss
        """
        H = self.conf.NSTEPS  # Actor prediction horizon
        n = self.conf.NSTEPS_TD_N  # TD bootstrap steps
        gamma = self.conf.gamma
        B = states.shape[0]

        # Actor outputs H actions per state: (B, H, A)
        actions_seq = self.eval(actor, states, is_actor=True)  # (B, H, A)

        # Use only first n actions for computing multi-step return
        actions_seq_n = actions_seq[:, :n]

        # Prepare containers for rollout and rewards
        rewards_seq = []
        current_states = states  # (B, D)

        for h in range(n):
            actions_h = actions_seq_n[:, h]  # (B, A)

            rewards_h = self.env.reward_batch(current_states, actions_h)  # (B,)
            rewards_seq.append(rewards_h)

            current_states = self.env.simulate_batch(current_states, actions_h)  # (B, D)

        final_states = current_states  # s_{t+n}
        V_final = self.eval(critic, final_states, is_actor=False).squeeze(-1)  # (B,)

        # Compute discounted rewards sum for n steps
        rewards_seq = torch.stack(rewards_seq, dim=1).squeeze(-1)  # (B, n)
        discounts = gamma ** torch.arange(n, device=states.device, dtype=torch.float32)  # (n,)
        returns = (rewards_seq * discounts.view(1, -1)).sum(dim=1)  # (B,)
        returns = returns.view(-1)     # Ensure shape [B]
        V_final = V_final.view(-1)     # Ensure shape [B]
        returns += (gamma ** n) * V_final

        # Loss = negative return
        loss = -returns.mean()
        actor.zero_grad()
        loss.backward()
        return loss.item()