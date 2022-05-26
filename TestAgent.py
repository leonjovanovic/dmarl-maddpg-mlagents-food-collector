from collections import deque
import os
import numpy as np
import torch.cuda
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

import Config
from NN import PolicyNN


class TestAgent:
    def __init__(self, env, behavior_name, num_agents, state_shape, action_cont_shape, action_disc_shape):
        self.num_agents = num_agents
        self.behavior_name = behavior_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = []
        for i in range(num_agents):
            self.policy_nn.append(
                PolicyNN(state_shape[0] * state_shape[1] * state_shape[2], action_cont_shape, action_disc_shape).to(
                    self.device))
        self.env = env
        self.ep_reward_agents = [0] * (Config.num_of_envs * Config.num_of_agents)
        self.return_queue = deque(maxlen=500)

    def test(self, nns, writer, n_step):
        flag = True
        # Create new enviroment and test it for 100 episodes using the model we trained
        self.test_reset(nns)
        decision_steps, terminal_steps = self.get_steps()
        n_episode = 0
        print("Testing...")
        print("Episodes finished ", end="")
        while n_episode < (Config.test_episodes / Config.num_of_envs):
            # Get the action from Policy NN given the state
            state_t = torch.flatten(torch.Tensor(decision_steps.obs[0]).to(self.device), start_dim=1)
            action_cont = torch.zeros((decision_steps.obs[0].shape[0], 3)).to(self.device)
            action_disc = torch.zeros((decision_steps.obs[0].shape[0], 1)).to(self.device)
            for i in range(Config.num_of_agents):
                action_cont[i * Config.num_of_envs: (i + 1) * Config.num_of_envs, :], action_disc[i * Config.num_of_envs: (i + 1) * Config.num_of_envs, :] = self.policy_nn[i](state_t[i * Config.num_of_envs: (i + 1) * Config.num_of_envs])
            if flag:
                flag = False
                #print(action_cont.detach().cpu().numpy())
                #print(action_disc.detach().cpu().numpy())
            actionsTuple = ActionTuple(discrete=action_disc.detach().cpu().numpy(),
                                       continuous=action_cont.detach().cpu().numpy())
            self.env.set_actions(self.behavior_name, actionsTuple)
            self.env.step()
            decision_steps, terminal_steps = self.get_steps()

            for a_id in decision_steps.agent_id:
                self.ep_reward_agents[a_id] += decision_steps.reward[a_id]

            if len(terminal_steps) > 0:
                for a_id in terminal_steps.agent_id:
                    self.return_queue.append(self.ep_reward_agents[a_id])
                self.ep_reward_agents = [0] * (Config.num_of_envs * Config.num_of_agents)
                self.env.reset()
                decision_steps, terminal_steps = self.get_steps()
                n_episode += 1
                print(str(n_episode * 4) + " ", end="")
        print("!")
        print(self.return_queue)
        mean_return = np.mean(self.return_queue)
        self.env.reset()
        if writer is not None:
            writer.add_scalar('test100rew', mean_return, n_step)
        return self.check_goal(mean_return)

    def test_reset(self, nns):
        for i in range(self.num_agents):
            self.policy_nn[i].load_state_dict(nns[i].state_dict())
        self.env.reset()

    def get_steps(self):
        steps = list(self.env.get_steps(self.behavior_name))
        return steps[0], steps[1]

    def check_goal(self, mean_return):
        for i in range(Config.num_of_agents):
            if os.path.exists('models/FoodCollector' + str(i) + "_" + Config.date_time + '.pt'):  # checking if there is a file with this name
                os.remove('models/FoodCollector' + str(i) + "_" + Config.date_time + '.pt')  # deleting the file
            torch.save(self.policy_nn[i].state_dict(), 'models/FoodCollector' + "_" + Config.date_time + "_" + str(i) + '.pt')  # saving a new model with the same name
        if mean_return < 10:
            print("Goal NOT reached! Mean 100 test reward: " + str(np.round(mean_return, 2)))
            return False
        else:
            print("GOAL REACHED! Mean reward over 100 episodes is " + str(np.round(mean_return, 2)))
            # If we reached goal, save the model locally
            #for i in range(Config.num_of_agents):
            #    torch.save(self.policy_nn[i].state_dict(), 'models/FoodCollector' + str(i) + "_" + Config.date_time + '.pt')
            return True
