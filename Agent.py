import itertools
from collections import deque
import numpy as np
from tensorboardX import SummaryWriter

import Config
from AgentControl import AgentControl
from Buffer import Buffer
from TestAgent import TestAgent


class Agent:
    def __init__(self, env, behavior_name, state_shape, action_cont_shape, action_disc_shape, num_agents):
        self.agent_control = AgentControl(state_shape, action_cont_shape, action_disc_shape, num_agents)
        self.buffer = Buffer(state_shape, action_cont_shape + 1)
        self.test_agent = TestAgent(env, behavior_name, num_agents, state_shape, action_cont_shape, action_disc_shape)
        self.writer = SummaryWriter(logdir='content/runs/' + Config.writer_name) if Config.write else None
        self.policy_loss_mean = deque(maxlen=100)
        self.critic_loss_mean = deque(maxlen=100)
        self.return_queue = deque(maxlen=100)
        self.ep_reward_agents = [0] * (Config.num_of_envs * Config.num_of_agents)
        self.max_reward = -100

    def get_actions(self, state, n_step):
        if n_step < Config.start_steps:
            action_cont, action_disc = self.agent_control.get_actions_random(state)
            #_, _ = self.agent_control.get_actions(state, n_step, self.buffer.buffer_index)
        else:
            action_cont, action_disc = self.agent_control.get_actions(state, n_step, self.buffer.buffer_index)
        self.buffer.add_first_part(state, action_cont, action_disc)
        return action_cont, action_disc

    @staticmethod
    def get_steps(env, behavior_name):
        steps = list(env.get_steps(behavior_name))
        return steps[0], steps[1]

    def calculate_ep_reward(self, decision_steps):
        for a_id in decision_steps.agent_id:
            self.ep_reward_agents[a_id] += decision_steps.reward[a_id]

    def add_to_buffer(self, decision_steps, terminal_steps):
        self.buffer.add_second_part(decision_steps, terminal_steps)

    def update(self, n_step):
        self.agent_control.lr_std_decay(n_step)
        if self.buffer.buffer_index < Config.min_buffer_size and not self.buffer.initialized:
            return
        indices = self.buffer.sample_indices()
        critic_losses = self.agent_control.critic_update(self.buffer.states[indices], self.buffer.actions[indices], self.buffer.rewards[indices], self.buffer.new_states[indices], self.buffer.dones[indices])
        policy_losses = self.agent_control.policy_update(self.buffer.states[indices], self.buffer.actions[indices])
        self.agent_control.target_update()
        self.critic_loss_mean.append(np.mean(np.array(critic_losses)))
        self.policy_loss_mean.append(np.mean(np.array(policy_losses)))

    def record_data(self, n_step):
        #if (n_step + 1) % Config.episode_length != 0:  # or self.buffer.buffer_index < Config.min_buffer_size:
        #    return
        self.max_reward = np.maximum(self.max_reward, np.max(self.ep_reward_agents))
        print("St " + str(n_step) + "/" + str(Config.total_steps) + " Mean 100 policy loss: " + str(
            np.round(np.mean(self.policy_loss_mean), 4)) + " Mean 100 critic loss: " + str(
            np.round(np.mean(self.critic_loss_mean), 4)) + " Max reward: " + str(
            np.round(self.max_reward, 2)) + " Mean 100 reward: " + str(
            np.round(np.mean(self.return_queue), 2)) + " Last rewards: " + str(
            np.round(self.ep_reward_agents, 2)))
        if Config.write and self.buffer.buffer_index > Config.min_buffer_size:
            self.writer.add_scalar('pg_loss', np.mean(self.policy_loss_mean), n_step)
            self.writer.add_scalar('vl_loss', np.mean(self.critic_loss_mean), n_step)
            self.writer.add_scalar('100rew', np.mean(self.return_queue), n_step)
            self.writer.add_scalar('rew', np.mean(self.ep_reward_agents), n_step)

    def reset(self, env, terminal_steps, n_step):
        for a_id in terminal_steps.agent_id:
            self.ep_reward_agents[a_id] += terminal_steps.reward[a_id]
            self.return_queue.append(self.ep_reward_agents[a_id])

        self.record_data(n_step)

        self.ep_reward_agents = [0] * (Config.num_of_envs * Config.num_of_agents)
        env.reset()

    def check_goal(self, n_step):
        if (n_step + 1) % Config.test_every == 0 or (
                len(self.return_queue) >= 100 and
                np.mean(list(itertools.islice(self.return_queue, 75, 100))) >= 10):
            return True
        return False

    def test(self, n_step):
        return self.test_agent.test(self.agent_control.moving_policy_nn, self.writer, n_step)



