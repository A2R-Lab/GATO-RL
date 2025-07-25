import torch
import numpy as np

def de_normalize_tensor(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
    state_time = torch.cat([torch.zeros([state.shape[0], state.shape[1] - 1]), torch.reshape((state[:, -1] + 1) * state_norm_arr[-1] / 2, (state.shape[0], 1))], dim=1)
    state_no_time = state * state_norm_arr
    mask = torch.cat([torch.ones([state.shape[0], state.shape[1] - 1]), torch.zeros([state.shape[0], 1])], dim=1)
    state_not_norm = state_no_time * mask + state_time * (1 - mask)
    
    return state_not_norm

def normalize_tensor(state, state_norm_arr):
    ''' Retrieve state from normalized state - tensor '''
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
