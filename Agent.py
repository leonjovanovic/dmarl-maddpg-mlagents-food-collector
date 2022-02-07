from AgentControl import AgentControl
from Buffer import Buffer


class Agent:
    def __init__(self, state_shape, action_cont_shape, action_disc_shape):
        self.state_shape = state_shape
        self.agent_control = AgentControl(state_shape, action_cont_shape, action_disc_shape)
        self.buffer = Buffer(state_shape, action_cont_shape + 1)

    @staticmethod
    def get_steps(env, behavior_name):
        steps = list(env.get_steps(behavior_name))
        return steps[0], steps[1]

    def get_actions(self, state):
        action_cont, action_disc = self.agent_control.get_actions(state)
        self.buffer.add_first_part(state, action_cont, action_disc)
        return action_cont, action_disc

    def add_to_buffer(self, decision_steps, terminal_steps):
        self.buffer.add_second_part(decision_steps, terminal_steps)


