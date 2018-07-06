import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.ac = env.action_space

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.ac.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		sampled_acts = np.array([[self.env.action_space.sample() for j in range(self.num_simulated_paths)] for i in range(self.horizon)])
		states = [np.array([state] * self.num_simulated_paths)]
		nstates = []

		for i in range(self.horizon):
			nstates.append(self.dyn_model.predict(states[-1], sampled_acts[i, :]))
			if i < self.horizon: states.append(nstates[-1])

		costs = trajectory_cost_fn(self.cost_fn, states, sampled_acts, nstates)
		return sampled_acts[0][np.argmin(costs)]
