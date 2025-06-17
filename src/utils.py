import torch
import numpy as np

'''
Note: the only 2 functions from this file that are called in main.py are array2tensor and normalize_tensor.
Therefore these were the only 2 functions that were changed and tested.
'''

def array2tensor(array):
    if isinstance(array, list):
        array = np.array(array)
    elif torch.is_tensor(array):
        return array
    return torch.unsqueeze(torch.tensor(array, dtype=torch.float32), 0)

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

def de_normalize(state, state_norm_arr):
    ''' Retrieve state from normalized state '''
    state_not_norm  = np.empty_like(state)
    state_not_norm[:-1] = state[:-1] * state_norm_arr[:-1]
    state_not_norm[-1] = (state[-1] + 1) * state_norm_arr[-1]/2

    return state_not_norm

def normalize(state, state_norm_arr):
    ''' Normalize state '''
    state_norm  = np.empty_like(state)
    state_norm = state / state_norm_arr
    state_norm[-1] = state_norm[-1] * 2 -1

    return state_norm
