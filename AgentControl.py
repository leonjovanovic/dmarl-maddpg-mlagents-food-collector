import numpy as np
import torch

from NN import PolicyNN


class AgentControl:
    def __init__(self, state_shape, action_cont_shape, action_disc_shape):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(state_shape[0] * state_shape[1] * state_shape[2], action_cont_shape, action_disc_shape).to(self.device)
        self.noise_std = 0.1

    def get_actions(self, state):
        action_cont, action_disc_prob = self.policy_nn(torch.flatten(torch.Tensor(state).to(self.device), start_dim=1))
        noise = (self.noise_std ** 0.5) * torch.randn(action_cont.shape).to(self.device)
        #Razlika izmedju verovatnoce i generisanog broja od 0 do 1
        choices = action_disc_prob.detach().cpu().numpy()[:, :1] - np.random.random((20, 1))
        # Veci jednak od 0 => 1, manji od nule => 0
        action_disc = np.array(np.greater(choices, 0), dtype=int)
        action_cont = torch.clip(action_cont + noise, -1, 1).detach().cpu().numpy()
        return action_cont, action_disc
