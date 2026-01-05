# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .memory import ReplayMemory, Transition
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size=36, action_size=9):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_size)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_size=36, action_size=9, replay_memory_size=10000, batch_size=32, gamma=0.99, learning_rate=1e-4, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.update_target_network()
        
        self.memory = ReplayMemory(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_update = target_update  # How often to update target network
        self.steps_done = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 500
        self.epsilon = self.epsilon_start
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def get_valid_actions(self, state):
        # return (state[:27][1::3] == 1)   # Empty cells in the original state
        return np.argwhere(state[:9] == 0).flatten()
        
    def select_action(self, state):
        rand = np.random.random()
        state = np.array(state)
        if rand > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                q_values = q_values.cpu().numpy()[0]
                # # Mask invalid actions
                valid_actions = self.get_valid_actions(state)
                valid_actions_mask = np.isin(np.arange(len(q_values)), valid_actions)
                q_values[~valid_actions_mask] = -np.inf
                action = np.argmax(q_values)
                return action
        else:
            # Randomly select from valid actions
            valid_actions = self.get_valid_actions(state)
            if len(valid_actions) == 0:
                return np.random.randint(0, self.action_size)
            return np.random.choice(valid_actions)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Unpack the batch
        batch = Transition(*zip(*transitions))
        
        # Prepare batches
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            next_q_values[done_batch == 1] = 0.0
        
        # Compute the expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss
        loss = F.mse_loss(q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (optional)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
    def update_epsilon(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)
    
    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)
        
    def load(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.update_target_network()

    def eval(self):
        self.epsilon = 0
        
    def select_top3_action(self, state):
        state = np.array(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            q_values = q_values.cpu().numpy()[0]
            # # Mask invalid actions
            valid_actions = self.get_valid_actions(state)
            valid_actions_mask = np.isin(np.arange(len(q_values)), valid_actions)
            q_values[~valid_actions_mask] = -np.inf
            top_actions = np.argsort(q_values)[-3:][::-1]
            return top_actions
    
    def get_q_values(self, state):
        state = np.array(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            q_values = q_values.cpu().numpy()[0]
            # # Mask invalid actions
            valid_actions = self.get_valid_actions(state)
            valid_actions_mask = np.isin(np.arange(len(q_values)), valid_actions)
            q_values[~valid_actions_mask] = -np.inf
            action = np.argmax(q_values)
            return action, q_values