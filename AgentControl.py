import numpy as np
import torch

import Config
from NN import PolicyNN, CriticNN


class AgentControl:
    def __init__(self, state_shape, action_cont_shape, action_disc_shape, num_agents):
        self.action_cont_shape = action_cont_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.moving_policy_nn = []
        self.moving_critic_nn = []
        self.policy_nn_optim = []
        self.critic_nn_optim = []
        self.target_policy_nn = []
        self.target_critic_nn = []
        for i in range(num_agents):
            self.moving_policy_nn.append(
                PolicyNN(state_shape[0] * state_shape[1] * state_shape[2], action_cont_shape, action_disc_shape).to(
                    self.device))
            self.policy_nn_optim.append(
                torch.optim.Adam(self.moving_policy_nn[i].parameters(), lr=Config.policy_lr, eps=Config.adam_eps))
            self.moving_critic_nn.append(CriticNN(
                num_agents * state_shape[0] * state_shape[1] * state_shape[2] + num_agents * (
                            action_cont_shape + 1)).to(self.device))
            self.critic_nn_optim.append(
                torch.optim.Adam(self.moving_critic_nn[i].parameters(), lr=Config.critic_lr, eps=Config.adam_eps))
            self.target_policy_nn.append(
                PolicyNN(state_shape[0] * state_shape[1] * state_shape[2], action_cont_shape, action_disc_shape).to(
                    self.device))
            self.target_critic_nn.append(CriticNN(
                num_agents * state_shape[0] * state_shape[1] * state_shape[2] + num_agents * (
                            action_cont_shape + 1)).to(self.device))
        self.mse = torch.nn.MSELoss()
        self.noise_std = 0.1

    def get_actions(self, state):
        # Transform 20x40x40x5 to 4x5x8000
        state_t = torch.flatten(torch.Tensor(state).to(self.device), start_dim=1)
        # NN output will be 1x3 and 1x2, we need to stack them to 20x3 and 20x2
        # action_cont = torch.zeros((state.shape[0], 3)).to(self.device)
        # action_disc_prob = torch.zeros((state.shape[0], 2)).to(self.device) # 2
        actions = torch.zeros((state.shape[0], 4)).to(self.device)
        for i in range(Config.num_of_envs * Config.num_of_agents):
            actions[i, :] = self.moving_policy_nn[i % Config.num_of_agents](state_t[i, :])
        noise = (self.noise_std ** 0.5) * torch.randn((state.shape[0], 4)).to(self.device)
        actions = torch.clip(actions + noise, -1, 1).detach().cpu().numpy()
        # Razlika izmedju generisanog broja od 0 do 1 i verovatnoce
        # choices = np.random.random((state.shape[0], 1)) - action_disc_prob.detach().cpu().numpy()[:, :1]
        # Veci jednak od 0 => 1, manji od nule => 0
        # action_disc = np.array(np.greater(choices, 0), dtype=int)
        return actions[:, :3], actions[:, 3:]

    def get_actions_random(self, state):
        #action_cont = np.random.random((state.shape[0], self.action_cont_shape)) * 2 - 1
        #action_disc = np.round(np.random.random((state.shape[0], 1)))
        actions = np.random.random((state.shape[0], self.action_cont_shape + 1)) * 2 - 1
        return actions[:, :3], actions[:, 3:]

    def lr_std_decay(self, n_step):
        frac = 1 - n_step / Config.total_steps
        for i in range(Config.num_of_agents):
            self.policy_nn_optim[i].param_groups[0]["lr"] = frac * Config.policy_lr
            self.critic_nn_optim[i].param_groups[0]["lr"] = frac * Config.critic_lr
        self.noise_std = self.noise_std * frac

    def critic_update(self, states, actions, rewards, new_states):
        # States shape = batch_size x num_of_agents x 8000 (e.g.64x5x8000), action shape = batch_size x num_of_agents x 4
        # CriticNN input shape = num_of_agents x 8000 + num_of_agents x 4
        # state_f & new_state_f shape = 64 x 40000, action_f & new_action_f shape = 64 x 20
        states_f = torch.flatten(states, start_dim=1)
        actions_f = torch.flatten(actions, start_dim=1)
        new_states_f = torch.flatten(new_states, start_dim=1)
        new_actions_f = torch.zeros(actions_f.shape).to(self.device)
        for i in range(Config.num_of_agents):
            action_cont, action_disc_prob = self.target_policy_nn[i](new_states[:, i, :])
            choices = np.random.random((action_disc_prob.shape[0], 1)) - action_disc_prob.detach().cpu().numpy()[:, :1]
            action_disc = np.array(np.greater(choices, 0), dtype=int)
            action_cont = action_cont.detach()
            new_action = torch.cat((action_cont, torch.Tensor(action_disc).to(self.device)), dim=1)
            new_actions_f[:, i * new_action.shape[1]: (i + 1) * new_action.shape[1]] = new_action
        # Input for Target CriticNN = new state and new action (DETACHED), input for Moving CriticNN = state and action
        critic_losses = []
        for i in range(Config.num_of_agents):
            # NN output shape = batch_size x 1
            state_values = self.moving_critic_nn[i](states_f, actions_f)
            new_state_values = self.target_critic_nn[i](new_states_f, new_actions_f).detach()
            # rewards shape = batch_size x 1, CriticNN output shape = batch_size x 1
            target = rewards[:, i] + Config.gamma * new_state_values
            critic_losses.append(self.mse(state_values, target))

            self.critic_nn_optim[i].zero_grad()
            critic_losses[i].backward()
            self.critic_nn_optim[i].step()
            critic_losses[i] = critic_losses[i].detach()
        return critic_losses

    def policy_update(self):
        pass
