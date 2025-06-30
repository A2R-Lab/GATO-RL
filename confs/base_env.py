import numpy as np
import torch

class BaseEnv:
	"""
	Base environment class defining the interface and utility functions
	for dynamics, cost, and reward. This class provides functions mainly used for
	computing the reward-to-go (rl_trainer.compute_partial_rtg()) and the actor
	network update (neural_network.compute_actor_grad()). Also, to create random
	initial states at each episode for TO to solve.

	Specific environments (e.g. Pendulum) should subclass this and override
	environment-specific methods.
	"""
	def __init__(self, conf):
		self.conf = conf

	def reset_batch(self, batch_size):
		"""
		Reset the environment to a random initial state for a batch of size `batch_size`.

		Args:
			batch_size (int): Number of initial states to reset

		Returns:
			np.ndarray: Array of shape (batch_size, nx+1), where each row is the state
						appended by the timestep
		"""
		raise NotImplementedError
	
	def simulate(self, state, action):
		"""
		Get the next state upon applying the action.

		References:
			rl_trainer.compute_partial_rtg()

		Args:
			state (np.ndarray): Current state (nx+1,)
			action (np.ndarray): Action (na,)

		Returns:
			np.ndarray: Next state with shape (nx+1,)
		"""
		raise NotImplementedError
	
	def simulate_batch(self, state, action):
		"""
		Batch version of simulate().

		References:
			neural_network.compute_actor_grad()

		Args:
			state (torch.tensor): Batch of states with shape (batch_size, nx+1)
			action (torch.tensor): Batch of actions with shape (batch_size, na)

		Returns:
			torch.Tensor: Batch of next states with shape (batch_size, nx+1)
		"""
		raise NotImplementedError

	def derivative_batch(self, state, action):
		"""
		Batch version of derivative()

		References:
			neural_network.compute_actor_grad()

		Args:
			state (torch.Tensor): Batch of states (batch_size, nx+1)
			action (torch.Tensor): Batch of actions (batch_size, na)

		Returns:
			torch.Tensor: Batch of derivative matrices (batch_size, nx+1, na)
		"""
		raise NotImplementedError

	def reward(self, state, action=None):
		"""
		Computes reward from state and action.

		References:
			rl_trainer.compute_partial_rtg()

		Args:
			state (torch.Tensor): Current state.
			action (torch.Tensor, optional): Control input.

		Returns:
			float: Reward value.
		"""
		raise NotImplementedError

	def reward_batch(self, state_batch, action_batch=None):
		"""
		Computes batch of rewards using tensors.

		References:
			neural_network.compute_actor_grad()

		Args:
			state_batch (torch.Tensor): Batch of states (batch_size, nx+1)
			action_batch (torch.Tensor, optional): Batch of actions (batch_size, na)

		Returns:
			torch.Tensor: Batch of reward values (batch_size,)
		"""
		raise NotImplementedError