import sys
import os
import numpy as np
import torch
import pytest
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
conf = importlib.import_module('confs.pendulum_conf')
PendulumEnv = getattr(conf, 'PendulumEnv')

@pytest.fixture
def env():
	return PendulumEnv(conf, N_ts=400, u_min=5, u_max=5)

def test_reset_batch_shape(env):
	batch_size = 5
	states = env.reset_batch(batch_size)
	assert isinstance(states, np.ndarray)
	assert states.shape == (batch_size, 3)

def test_simulate_batch_shape(env):
	batch_size = 5
	states = np.random.rand(batch_size, 3).astype(np.float32)
	actions = np.random.rand(batch_size, 1).astype(np.float32)
	next_states = env.simulate_batch(torch.tensor(states), torch.tensor(actions))
	assert isinstance(next_states, torch.Tensor)
	assert next_states.shape == (batch_size, 3)

def test_derivative_batch_shape(env):
	batch_size = 4
	states = np.random.rand(batch_size, 3).astype(np.float32)
	actions = np.random.rand(batch_size, 1).astype(np.float32)
	jac = env.derivative_batch(states, actions)
	assert isinstance(jac, torch.Tensor)
	assert jac.shape == (batch_size, 3, 1)

def test_reward_scalar(env):
	state = np.array([0.1, 0.0, 0.0])
	action = np.array([0.1])
	r = env.reward(state, action)
	assert isinstance(r, float)

def test_reward_batch_shape(env):
	batch_size = 6
	states = torch.rand(batch_size, 3)
	actions = torch.rand(batch_size, 1)
	rewards = env.reward_batch(states, actions)
	assert isinstance(rewards, torch.Tensor)
	assert rewards.shape == (batch_size, 1)
