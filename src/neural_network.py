import numpy as np
import torch
import torch.nn as nn
from siren_pytorch import Siren
from utils import normalize_tensor
    
class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, inputs, targets, weights=None):
        if inputs.shape != targets.shape:
            raise ValueError("Inputs and targets must have the same shape")
        
        mse_loss = torch.pow(inputs - targets, 2)
        
        if weights is not None:
            if weights.shape != inputs.shape:
                weights.expand(inputs.shape)
            mse_loss = mse_loss * weights
        return torch.mean(mse_loss)

class ActorCriticNet:
    def __init__(self, env, conf):
        self.env = env
        self.conf = conf
        self.MSE = WeightedMSELoss()
        self.batch_size = conf.BATCH_SIZE
        return

    def create_actor(self):
        model = nn.Sequential(
            nn.Linear(self.conf.nb_state, self.conf.NH1),
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

    def create_critic_elu(self): 
        model = nn.Sequential(
            nn.Linear(self.conf.nb_state, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        return model.to(torch.float32)
    
    def create_critic_sine_elu(self): 
        model = nn.Sequential(
            Siren(self.conf.nb_state, 64),
            nn.Linear(64, 64),
            nn.ELU(),
            Siren(64, 128),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128,1)
        )

        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        return model.to(torch.float32)
        
    def create_critic_sine(self): 
        model = nn.Sequential(
            Siren(self.conf.nb_state, 64),
            Siren(64, 64),
            Siren(64, 128),
            Siren(128, 128),
            nn.Linear(128, 1)
        )

        nn.init.xavier_uniform_(model[-1].weight)
        nn.init.constant_(model[-1].bias, 0)
        return model.to(torch.float32)
        
    def create_critic_relu(self): 
        model = nn.Sequential(
            nn.Linear(self.conf.nb_state, 16),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(16, 32),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(32, self.conf.NH1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH1, self.conf.NH2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(self.conf.NH2, 1)
        )
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        return model.to(torch.float32)
    
    def eval(self, NN, input):
        if not torch.is_tensor(input):
            if isinstance(input, list):
                input = np.array(input)
            input = torch.tensor(input, dtype=torch.float32)

        if self.conf.NORMALIZE_INPUTS:
            input = normalize_tensor(input, torch.tensor(self.conf.state_norm_arr, dtype=torch.float32))

        return NN(input)
     
    
    def compute_critic_grad(self, critic_model, target_critic, state_batch, 
                            state_next_rollout_batch, partial_reward_to_go_batch, d_batch,
                            weights_batch):
        # compute cost-to-go
        if self.conf.MC:
            reward_to_go_batch = partial_reward_to_go_batch
        else:
            reward_to_go_batch = partial_reward_to_go_batch +\
                (1 - d_batch) * self.eval(target_critic, state_next_rollout_batch)

        # compute mse between cost-to-go and V
        critic_value = self.eval(critic_model, state_batch)
        critic_loss = self.MSE(reward_to_go_batch, critic_value, weights=weights_batch)
        critic_model.zero_grad()
        critic_loss.backward()

        return reward_to_go_batch, critic_value, self.eval(target_critic, state_batch)

    def compute_actor_grad(self, actor_model, critic_model, state_batch):
        actions = self.eval(actor_model, state_batch)

        # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
        act_np = actions.detach().cpu().numpy()        
        state_next_tf, ds_next_da = self.env.simulate_batch(state_batch.detach().cpu().numpy(), act_np), self.env.derivative_batch(state_batch.detach().cpu().numpy(), act_np)
        state_next_tf = state_next_tf.clone().detach().to(dtype=torch.float32).requires_grad_(True)
        ds_next_da = ds_next_da.clone().detach().to(dtype=torch.float32).requires_grad_(True)

        # Compute critic value at the next state
        critic_value_next = self.eval(critic_model, state_next_tf)

        # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)
        dV_ds_next = torch.autograd.grad(outputs=critic_value_next, inputs=state_next_tf,
                                        grad_outputs=torch.ones_like(critic_value_next),
                                        create_graph=True)[0]

        # Compute rewards
        state_batch_np = state_batch.detach().cpu().numpy()
        rewards_tf = self.env.reward_batch(state_batch, actions)

        # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
        dr_da = torch.autograd.grad(outputs=rewards_tf, inputs=actions,
                                    grad_outputs=torch.ones_like(rewards_tf),
                                    create_graph=True)[0]

        dr_da_reshaped = dr_da.view(self.batch_size, 1, self.conf.na)

        # dr_ds' + dV_ds' (note: dr_ds' = 0)
        dQ_ds_next = dV_ds_next.view(self.batch_size, 1, self.conf.nb_state)

        # (dr_ds' + dV_ds')*ds'_da
        dQ_ds_next_da = torch.bmm(dQ_ds_next, ds_next_da)

        # (dr_ds' + dV_ds')*ds'_da + dr_da
        dQ_da = dQ_ds_next_da + dr_da_reshaped

        # Multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a
        actions = self.eval(actor_model, state_batch)
        actions_reshaped = actions.view(self.batch_size, self.conf.na, 1)
        dQ_da_reshaped = dQ_da.view(self.batch_size, 1, self.conf.na)
        #Q_neg = torch.bmm(-dQ_da_reshaped, actions_reshaped)
        Q_neg = torch.matmul(-dQ_da_reshaped, actions_reshaped)

        # Compute the mean -Q across the batch
        mean_Qneg = Q_neg.mean()
        total_loss = mean_Qneg #+ self.compute_reg_loss(actor_model, True)

        # Gradients of the actor loss w.r.t. actor's parameters
        actor_model.zero_grad()
        #actor_grad = torch.autograd.grad(mean_Qneg, actor_model.parameters())
        total_loss.backward()
        # for param in actor_model.parameters():
        #     if param.grad is not None:
        #         param.grad.data /= 10
        #actor_grad = [param.grad for param in actor_model.parameters()]
        #print()
        #return actor_grad

    def compute_reg_loss(self, model, actor):
        #NOTE: layers in the original tf code were using kreg_l2_C (from self.conf) for all regularization parameters. 
        #This doesn't make sense and was changed here. Also, the original codebase used the keras 
        #bias_regularizer and kernel_regularizer variables, but never accessed the actor_model.losses
        #parameter to actually use the regularization loss in gradient computations.
        #I ended up not using this since it caused issues
        reg_loss = 0
        kreg_l1 = 0
        kreg_l2 = 0
        breg_l1 = 0
        breg_l2 = 0
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
                    l1_regularization_w = kreg_l1 * torch.sum(torch.abs(layer.weight))
                    reg_loss += l1_regularization_w
                if kreg_l2 > 0:
                    l2_regularization_w = kreg_l2 * torch.sum(torch.pow(layer.weight, 2))
                    reg_loss += l2_regularization_w

                # Regularization for biases
                if breg_l1 > 0:
                    l1_regularization_b = breg_l1 * torch.sum(torch.abs(layer.bias))
                    reg_loss += l1_regularization_b
                if breg_l2 > 0:
                    l2_regularization_b = breg_l2 * torch.sum(torch.pow(layer.bias, 2))
                    reg_loss += l2_regularization_b
        return reg_loss
