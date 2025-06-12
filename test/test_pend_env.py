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

def test_reset_batch_bounds(env):
	batch_size = 5
	states = env.reset_batch(batch_size)
	x_min = env.conf.x_init_min
	x_max = env.conf.x_init_max
	assert np.all(states >= x_min)
	assert np.all(states <= x_max)

def test_simulate_batch_shape(env):
	batch_size = 5
	states = np.random.rand(batch_size, 3).astype(np.float32)
	actions = np.random.rand(batch_size, 1).astype(np.float32)
	next_states = env.simulate_batch(torch.tensor(states), torch.tensor(actions))
	assert isinstance(next_states, torch.Tensor)
	assert next_states.shape == (batch_size, 3)

def test_simulate_batch_equilibrium(env):
	batch_size = 5
	states = torch.tensor([[0.0, 0.0, 0.0]] * batch_size, dtype=torch.float32)
	actions = torch.zeros((batch_size, 1))
	next_states = env.simulate_batch(states, actions)
	assert torch.allclose(next_states[:, :2], states[:, :2], atol=1e-4)

def test_simulate_batch(env):
	batch_size = 1
	state = torch.tensor([[np.pi, 0.0, 0.0]], dtype=torch.float32)
	action = torch.tensor([[1.0]], dtype=torch.float32)
	next_state = env.simulate_batch(state, action)
	next_theta = next_state[0, 0].item()
	next_omega = next_state[0, 1].item()

	# Expected results based on Euler integration
	expected_theta = np.pi # θ remains the same
	expected_omega = env.conf.dt # ω = dt * (u - g * sin(π)) = dt * 1.0

	assert np.isclose(next_theta, expected_theta, atol=1e-6), f"Theta mismatch: {next_theta}"
	assert np.isclose(next_omega, expected_omega, atol=1e-6), f"Omega mismatch: {next_omega}"

def test_derivative_batch_shape(env):
	batch_size = 5
	states = np.random.rand(batch_size, 3).astype(np.float32)
	actions = np.random.rand(batch_size, 1).astype(np.float32)
	jac = env.derivative_batch(states, actions)
	assert isinstance(jac, torch.Tensor)
	assert jac.shape == (batch_size, 3, 1)

def test_derivative_batch(env):
	batch_size = 5
	states = np.random.rand(batch_size, 3).astype(np.float32)
	actions = np.random.rand(batch_size, 1).astype(np.float32)
	jac = env.derivative_batch(states, actions)

	# Only ω is affected by u, with ∂ω/∂u = dt
	expected = torch.zeros((batch_size, 3, 1))
	expected[:, 1, 0] = env.conf.dt
	assert torch.allclose(jac, expected, atol=1e-6), f"Unexpected derivative: {jac}"

def test_reward_scalar(env):
	state = np.array([0.1, 0.0, 0.0])
	action = np.array([0.1])
	r = env.reward(state, action)
	assert isinstance(r, float)

def test_reward_penalizes_deviation(env):
	goal = env.conf.goal_state
	goal_theta = goal[0]
	goal_omega = goal[1]
	state_near_goal = np.array([goal_theta, goal_omega, 0.0])
	state_far = np.array([goal_theta + 1.0, goal_omega + 1.0, 0.0])
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
	assert r_small > r_large, "Reward should be lower for larger torque inputs"

def test_reward_scalar_matches_cost(env):
	# Create 1-step trajectory
	theta, omega, torque = np.pi, 0.5, 1.0
	state = np.array([theta, omega, torque])
	action = np.array([1.0])

	# Compute reward and cost from single-step trajectory
	reward = env.reward(state, action)
	env.N = 1
	cost = env.running_cost(np.array([theta, omega, torque]))
	assert np.isclose(reward, -cost, atol=1e-6), f"Reward and running cost mismatch: {reward} != {-cost}"

def test_reward_batch_matches_scalar(env):
	states = torch.tensor([[0.0, 0.0, 0.0],
							[np.pi, 0.0, 0.0]], dtype=torch.float32)
	actions = torch.tensor([[0.0],[0.5]], dtype=torch.float32)
	rewards_batch = env.reward_batch(states, actions).squeeze()
	for i in range(states.shape[0]):
		r_scalar = env.reward(states[i].numpy(), actions[i].numpy())
		assert np.isclose(rewards_batch[i].item(), r_scalar, atol=1e-6), \
			f"Mismatch between batch and scalar reward at index {i}"

def test_reward_batch_shape(env):
	batch_size = 6
	states = torch.rand(batch_size, 3)
	actions = torch.rand(batch_size, 1)
	rewards = env.reward_batch(states, actions)
	assert isinstance(rewards, torch.Tensor)
	assert rewards.shape == (batch_size, 1)
