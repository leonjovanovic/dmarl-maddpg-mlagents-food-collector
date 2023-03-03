import torch
import torch.nn as nn
import torch.nn.functional as f

import Config

#torch.manual_seed(Config.seed)

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_cont_shape, output_disc_shape):
        super(PolicyNN, self).__init__()
        self.disc_shape = output_disc_shape
        self.model = nn.Sequential(
            nn.Linear(input_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_cont_shape + output_disc_shape)
        )
        self.cont = nn.ReLU()
        #self.disc = nn.Softmax(dim=-1)

    def forward(self, state):
        output = self.model(state)
        action_cont = output[..., :-self.disc_shape]#self.cont(output[..., :-self.disc_shape])
        action_disc = nn.functional.gumbel_softmax(output[..., -self.disc_shape:], hard=True, dim=-1)
        return action_cont, action_disc[..., -1:]

class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, states, actions):
        return self.model(torch.cat((states, actions), 1))
