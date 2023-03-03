import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import torch
from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

import Config
from Agent import Agent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# This is a non-blocking call that only loads the environment.
channel = StatsSideChannel()
msg = IncomingMessage(buffer=bytes(1000))
channel.on_message_received(msg)
env = UnityEnvironment(
    file_name='D:/Users/Leon Jovanovic/Documents/Computer Science/Unity Projects/ml-agents/Project/BuildFoodCollector/UnityEnvironment.exe',
    seed=1, side_channels=[channel])
# Start interacting with the environment.
env.reset()
behavior_name = next(iter(env.behavior_specs.keys()))
behavior_specs = next(iter(env.behavior_specs.values()))
print(behavior_name)
state_size = behavior_specs[0][0].shape
action_size_cont = behavior_specs[1].continuous_size
action_size_disc = behavior_specs[1].discrete_branches[0]
# NA POCETKU GLEDAMO KOME SVE TREBA AKCIJA
steps = list(env.get_steps(behavior_name))
decision_steps = steps[0]
terminal_steps = steps[1]
num_steps = len(decision_steps) + len(terminal_steps)
agent_ids = decision_steps.agent_id
# Jedan state je 40x40x5
# Jedna akcija je 3 continuous akcije (napred, strana i rotacija) + binarna akcija (pucaj ili ne)
print(agent_ids)
agent = Agent(env, behavior_name, state_size, action_size_cont, action_size_disc, Config.num_of_agents)
for n_step in range(Config.total_steps):
    # For printing and writing to TensorBoard purposes, accumulate reward each step of an episode.
    agent.calculate_ep_reward(decision_steps)

    action_cont, action_disc = agent.get_actions(decision_steps.obs[0], n_step)
    action = ActionTuple(discrete=action_disc, continuous=action_cont)
    env.set_actions(behavior_name, action)
    env.step()
    decision_steps, terminal_steps = agent.get_steps(env, behavior_name)
    agent.add_to_buffer(decision_steps, terminal_steps)
    agent.update(n_step)

    if len(terminal_steps) > 0:
        # Add terminal reward, record data(Tensorboard) and reset environment
        agent.reset(env, terminal_steps, n_step)
        # Check if the model is to be tested. If it is, test and retreive new required decisions because test reseted env.
        if agent.check_goal(n_step):
           if agent.test(n_step):
               break
        decision_steps, terminal_steps = agent.get_steps(env, behavior_name)

# tensorboard --logdir="D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\dmarl-ml-agents-food-collector\content\runs" --host=127.0.0.1
# 1,800,000