# random agent that selects actions randomly
import numpy as np


class RandomAgent:
    def __init__(self, one_hot_state = False):
        self.one_hot_state = one_hot_state
        self.steps_done = 0
        
    def select_action(self, state):
        if not self.one_hot_state:
            valid_actions = np.argwhere(state == 0).flatten()
            return np.random.choice(valid_actions)
        else:
            # TODO: implement one-hot state action selection for RandomAgent
            return None
    
    def agent_name(self):
        return "RandomAgent"
    
    def optimize_model(self):
        pass