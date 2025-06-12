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

    def create_actor(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, self.conf.NH1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH1, self.conf.NH2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH2, self.conf.na)
        )

        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        return model.to(torch.float32)

    def create_critic_sine(self):
        model = nn.Sequential(
            Siren(self.state_dim, 64),
            Siren(64, 64),
            Siren(64, 128),
            Siren(128, 128),
            nn.Linear(128, 1)
        )

        nn.init.xavier_uniform_(model[-1].weight)
        nn.init.constant_(model[-1].bias, 0)
        return model.to(torch.float32)

    def eval(self, NN, input):
        if not torch.is_tensor(input):
            input = torch.tensor(np.array(input), dtype=torch.float32)

        if self.conf.NORMALIZE_INPUTS:
            norm_arr = torch.tensor(self.conf.NORM_ARR, dtype=torch.float32)
            input = normalize_tensor(input, norm_arr)

        return NN(input)
    
    def compute_critic_grad(self, critic, target_critic, states, next_states,
                            partial_rtg, dones, weights):
        if self.conf.MC:
            full_rtg = partial_rtg
        else:
            # if n-step TD, add remaining reward-to-go using target critic
            next_vals = self.eval(target_critic, next_states)
            full_rtg = partial_rtg + (1 - dones) * next_vals

        values = self.eval(critic, states)
        loss = self.MSE(values, full_rtg)
        critic.zero_grad()
        loss.backward()

        return full_rtg, values, self.eval(target_critic, states)

    def compute_actor_grad(self, actor, critic, states):
        actions = self.eval(actor, states)
        states_np, actions_np = states.detach().cpu().numpy(), actions.detach().cpu().numpy()
        next_states = self.env.simulate_batch(states_np, actions_np)
        next_states = next_states.clone().detach().to(torch.float32).requires_grad_(True)

        # ds'/da
        ds_next_da = self.env.derivative_batch(states_np, actions_np)
        ds_next_da = ds_next_da.clone().detach().to(torch.float32).requires_grad_(True)

        # dV(s')/ds'
        V_next = self.eval(critic, next_states)
        dV_ds_next = torch.autograd.grad(
            outputs=V_next,
            inputs=next_states,
            grad_outputs=torch.ones_like(V_next),
            create_graph=True
        )[0].view(self.batch_size, 1, self.state_dim)

        # dR/da
        rewards = self.env.reward_batch(states, actions)
        dR_da = torch.autograd.grad(
            outputs=rewards,
            inputs=actions,
            grad_outputs=torch.ones_like(rewards),
            create_graph=True
        )[0].view(self.batch_size, 1, self.conf.na)

        # dQ/da = dR/da + dV(s')/ds' * ds'/da
        dQ_da = torch.bmm(dV_ds_next, ds_next_da) + dR_da
        dQ_da = dQ_da.view(self.batch_size, 1, self.conf.na)
        actions = self.eval(actor, states).view(self.batch_size, self.conf.na, 1)
        
        loss = torch.matmul(-dQ_da, actions).mean()
        actor.zero_grad()
        loss.backward()

    def compute_reg_loss(self, model, actor):
        reg_loss = 0
        if actor:
            kreg_l1 = self.conf.kreg_l1_A
            kreg_l2 = self.conf.kreg_l2_A
            breg_l1 = self.conf.breg_l1_A
            breg_l2 = self.conf.breg_l2_A
        else:
            kreg_l1 = self.conf.kreg_l1_C
            kreg_l2 = self.conf.kreg_l2_C
            breg_l1 = self.conf.breg_l1_C
            breg_l2 = self.conf.breg_l2_C

        for layer in model:
            if isinstance(layer, nn.Linear):
                if kreg_l1 > 0:
                    reg_loss += kreg_l1 * torch.sum(torch.abs(layer.weight))
                if kreg_l2 > 0:
                    reg_loss += kreg_l2 * torch.sum(torch.pow(layer.weight, 2))
                if breg_l1 > 0:
                    reg_loss += breg_l1 * torch.sum(torch.abs(layer.bias))
                if breg_l2 > 0:
                    reg_loss += breg_l2 * torch.sum(torch.pow(layer.bias, 2))
        return reg_loss