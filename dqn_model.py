# dqn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Minimal DeepMind-like CNN for Atari.
    Implements exactly:
    def __init__(self, input_channels=4, num_actions=4)
    def forward(self, x)
    """
    def __init__(self, input_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # compute conv output size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out = self._conv(dummy).view(1, -1).size(1)
        
        self.fc1 = nn.Linear(conv_out, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
