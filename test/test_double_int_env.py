import sys
import os
import numpy as np
import torch
import pytest
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
conf = importlib.import_module('confs.double_int_conf')
DoubleIntEnv = getattr(conf, 'DoubleIntegratorEnv')

@pytest.fixture
def env():
	return DoubleIntEnv(conf)

def test_reset_batch_shape(env):
	batch_size = 5
	states = env.reset_batch(batch_size)
	assert isinstance(states, np.ndarray)
	assert states.shape == (batch_size, 3)

def test_reset_batch_bounds(env):
	batch_size = 5
	states = env.reset_batch(batch_size)
	x_min = env.conf.X_INIT_MIN
	x_max = env.conf.X_INIT_MAX
	assert np.all(states >= x_min)
	assert np.all(states <= x_max)

def test_simulate_batch_shape(env):
	batch_size = 5
	states = torch.rand(batch_size, 3)
	actions = torch.rand(batch_size, 1)
	next_states = env.simulate_batch(states, actions)
	assert isinstance(next_states, torch.Tensor)
	assert next_states.shape == (batch_size, 3)

def test_simulate_batch_equilibrium(env):
	batch_size = 5
	states = torch.zeros(batch_size, 3, dtype=torch.float32)
	actions = torch.zeros(batch_size, 1, dtype=torch.float32)
	next_states = env.simulate_batch(states, actions)
	assert np.allclose(next_states[:, :2], states[:, :2], atol=1e-4)

def test_simulate_batch(env):
	# initial state of [0,0,0] with action of 1
	dt = env.conf.dt
	state = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
	action = torch.tensor([[1.0]], dtype=torch.float32)
	next_state = env.simulate_batch(state, action)

	next_p = next_state[0, 0].item()
	next_v = next_state[0, 1].item()
	next_t = next_state[0, 2].item()

	assert np.isclose(next_p, 0.0, atol=1e-6), f"Position mismatch: {next_pos}"
	assert np.isclose(next_v, dt, atol=1e-6), f"Velocity mismatch: {next_vel}"
	assert np.isclose(next_t, dt, atol=1e-6), f"Time mismatch: {next_time}"

def test_derivative_batch_shape(env):
	batch_size = 5
	states = torch.rand(batch_size, 3)
	actions = torch.rand(batch_size, 1)
	jac = env.derivative_batch(states, actions)
	assert isinstance(jac, torch.Tensor)
	assert jac.shape == (batch_size, 3, 1)

def test_derivative_batch(env):
	batch_size = 5
	states = torch.rand(batch_size, 3)
	actions = torch.rand(batch_size, 1)
	jac = env.derivative_batch(states, actions)

	# Only ω is affected by u, with ∂ω/∂u = dt
	expected = torch.zeros((batch_size, 3, 1))
	expected[:, 1, 0] = env.conf.dt
	assert torch.allclose(jac, expected, atol=1e-6), f"Unexpected derivative: {jac}"

def test_reward_penalizes_deviation(env):
	goal = env.conf.goal_state
	goal_p, goal_v = goal[0], goal[1]
	state_near_goal = np.array([goal_p, goal_v, 0.0])
	state_far = np.array([goal_p + 1.0, goal_v + 1.0, 0.0])
	action = np.array([0.0])
	r_near = env.reward(state_near_goal, action)
	r_far = env.reward(state_far, action)
	assert r_near > r_far, "Reward should be higher closer to goal"

def test_reward_penalizes_large_action(env):
	state = np.array([np.pi, 0.0, 0.0])
	small_action = np.array([0.1])
	large_action = np.array([10.0])
	r_small = env.reward(state, small_action)
	r_large = env.reward(state, large_action)
	assert r_small > r_large, "Reward should be lower for larger control inputs"

def test_reward_batch_shape(env):
	batch_size = 6
	states = torch.rand(batch_size, 3)
	actions = torch.rand(batch_size, 1)
	rewards = env.reward_batch(states, actions)
	assert isinstance(rewards, torch.Tensor)
	assert rewards.shape == (batch_size, 1)
