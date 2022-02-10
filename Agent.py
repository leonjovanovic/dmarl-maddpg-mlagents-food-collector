import numpy as np

import Config
from AgentControl import AgentControl
from Buffer import Buffer


class Agent:
    def __init__(self, state_shape, action_cont_shape, action_disc_shape, num_agents):
        self.state_shape = state_shape
        self.agent_control = AgentControl(state_shape, action_cont_shape, action_disc_shape, num_agents)
        self.buffer = Buffer(state_shape, action_cont_shape + 1)

    def get_actions(self, state, n_step):
        if n_step < Config.start_steps:
            action_cont, action_disc = self.agent_control.get_actions_random(state)
        else:
            action_cont, action_disc = self.agent_control.get_actions(state)
        self.buffer.add_first_part(state, action_cont, action_disc)
        return action_cont, action_disc

    @staticmethod
    def get_steps(env, behavior_name):
        steps = list(env.get_steps(behavior_name))
        return steps[0], steps[1]

    def add_to_buffer(self, decision_steps, terminal_steps):
        self.buffer.add_second_part(decision_steps, terminal_steps)

    def update(self, n_step):
        self.agent_control.lr_std_decay(n_step)
        if self.buffer.buffer_index < Config.min_buffer_size and not self.buffer.initialized:
            return
        indices = self.buffer.sample_indices()
        critic_losses = self.agent_control.critic_update(self.buffer.states[indices], self.buffer.actions[indices], self.buffer.rewards[indices], self.buffer.new_states[indices])
        #print(critic_losses)
        policy_losses = self.agent_control.policy_update(self.buffer.states[indices], self.buffer.actions[indices])
        #print(policy_losses)
        self.agent_control.target_update()



