import numpy as np
import torch.cuda

import Config


class Buffer:
    def __init__(self, state_shape, action_shape):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_cell_size = state_shape[0] * state_shape[1] * state_shape[2]
        self.action_cell_size = action_shape

        self.states = torch.zeros(Config.buffer_size, Config.num_of_agents, self.state_cell_size).to(self.device)
        self.new_states = torch.zeros(Config.buffer_size, Config.num_of_agents, self.state_cell_size).to(self.device)
        self.actions = torch.zeros(Config.buffer_size, Config.num_of_agents, action_shape).to(self.device)
        self.rewards = torch.zeros(Config.buffer_size, Config.num_of_agents).to(self.device)
        self.dones = torch.zeros(Config.buffer_size, Config.num_of_agents).to(self.device)

        self.buffer_index = 0
        self.initialized = False

    # We receive states and actions for all 5 x 4 agents, so we need to take 5 by 5
    # state is 20 x 2000, action is 20 x 4
    def add_first_part(self, state, action_cont, action_disc):
        state_t = torch.Tensor(state).to(self.device).view(Config.num_of_envs, Config.num_of_agents, self.state_cell_size)
        action_t = torch.cat([torch.Tensor(action_cont), torch.Tensor(action_disc)], dim=1).view(Config.num_of_envs, Config.num_of_agents, self.action_cell_size)

        self.states[self.buffer_index: self.buffer_index + Config.num_of_envs, :, :] = state_t
        self.actions[self.buffer_index: self.buffer_index + Config.num_of_envs, :, :] = action_t

    def add_second_part(self, decision_steps, terminal_steps):
        state_t = torch.Tensor(decision_steps.obs[0]).to(self.device).view(Config.num_of_envs, Config.num_of_agents, self.state_cell_size)
        self.new_states[self.buffer_index: self.buffer_index + Config.num_of_envs, :, :] = state_t
        reward_t = torch.Tensor(decision_steps.reward).to(self.device).view(Config.num_of_envs, Config.num_of_agents)
        self.rewards[self.buffer_index: self.buffer_index + Config.num_of_envs, :] = reward_t
        if len(terminal_steps) == 0:
            self.dones[self.buffer_index: self.buffer_index + Config.num_of_envs, :] = 0
        else:
            self.dones[self.buffer_index: self.buffer_index + Config.num_of_envs, :] = 1
        self.buffer_index = (self.buffer_index + Config.num_of_envs) % Config.buffer_size
        if self.buffer_index == 0 and not self.initialized:
            self.initialized = True

    def sample_indices(self):
        indices = np.arange(min(Config.buffer_size, self.buffer_index) if not self.initialized else Config.buffer_size)
        np.random.shuffle(indices)
        indices = indices[:Config.batch_size]
        return indices

