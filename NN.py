import torch
import torch.nn as nn
import torch.nn.functional as f

import Config

#torch.manual_seed(Config.seed)

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_cont_shape, output_disc_shape):
        super(PolicyNN, self).__init__()
        self.disc_shape = output_disc_shape
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64*5*5, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=output_cont_shape)

        self.relu = nn.ReLU()

        # self.model = nn.Sequential(
        #     nn.Linear(input_shape, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, output_cont_shape + output_disc_shape)
        # )
        # self.cont = nn.ReLU()
        #self.disc = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.conv1(state.permute(0, 3, 1, 2))
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = x.reshape(-1, 64*5*5)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x, 0
        output = self.model(state)
        action_cont = output[..., :-self.disc_shape]#self.cont(output[..., :-self.disc_shape])
        action_disc = nn.functional.gumbel_softmax(output[..., -self.disc_shape:], hard=True, dim=-1)
        return action_cont, action_disc[..., -1:]

class CriticNN(nn.Module):
    def __init__(self, input_shape):
        super(CriticNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64*5*5, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)

        self.relu = nn.ReLU()
        # self.model = nn.Sequential(
        #     nn.Linear(input_shape, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

    def forward(self, states, actions):
        x = self.conv1(states)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = x.view(-1, 64*5*5)
        
        x = self.fc1(torch.cat((x, actions), 1))
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x
        return self.model(torch.cat((states, actions), 1))
