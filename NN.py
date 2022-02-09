import torch
import torch.nn as nn

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_cont_shape, output_disc_shape):
        super(PolicyNN, self).__init__()
        self.disc_shape = output_disc_shape
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_cont_shape + output_disc_shape)
        )
        self.cont = nn.Tanh()
        self.disc = nn.Softmax(dim=-1)

    def forward(self, state):
        output = self.model(state)
        action_cont = self.cont(output[..., :-self.disc_shape])
        action_disc = self.disc(output[..., -self.disc_shape:])
        return action_cont, action_disc

class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, states, actions):
        return self.model(torch.cat((states, actions), 1))
