import random 
import numpy as np
from environment import Actions, Observations

class RandomActionAgent(object):
	def __init__(self, sticky_actions=False):
		self.valid_actions = {action.name for action in Actions}
		self.sticky_actions = sticky_actions # repeat actions
		self.prev_action = random.choice(list(self.valid_actions))
		self.num_repeat = 5
		self.repeat_count = 0
	
	def act(self, observation):
		
		# get the set of actions unavailable given the observation
		actions_not_available = {
			Actions(Observations.get_direction_by_index(obs_idx)).name 
			for obs_idx, obs in enumerate(observation)
			if obs==1
		}

		# get the names of the actions available by doing set subtraction
		actions_available = self.valid_actions - actions_not_available
		# only keep the actions available in Action(Enum) type
		actions_available = [Actions[name] for name in actions_available]
		if (self.sticky_actions) and (self.repeat_count % self.num_repeat != 0) and (self.prev_action is not None):
			action =  self.prev_action
		else: 
			action = random.choice(actions_available).name
		
		self.prev_action = action
		self.repeat_count += 1
		return  action