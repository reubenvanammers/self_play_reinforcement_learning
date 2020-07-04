from collections import namedtuple
import copy
import numpy as np
import torch
from anytree import NodeMixin
from torch import nn
from torch.functional import F
import random
from rl_utils.flat import MSELossFlat
from rl_utils.memory import Memory
from rl_utils.weights import init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNetTicTacToe(nn.Module):
    def __init__(self, width=3, height=3, action_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True)  # Deal with padding?
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + padding * 2 - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(width))), 1, 1, 0)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(height))), 1, 1, 0)
        linear_input_size = convw * convh

        # Policy Head
        self.conv_policy = nn.Conv2d(64, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_dropout = nn.Dropout(p=0.5)
        self.linear_policy = nn.Linear(linear_input_size * 2, action_size)
        self.softmax = nn.Softmax()

        # Value head
        self.conv_value = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_dropout = nn.Dropout(p=0.5)

        self.fc_value = nn.Linear(linear_input_size * 1, 256)
        self.linear_output = nn.Linear(256, 1)

        self.apply(init_weights)

    def __call__(self, state, player=1):
        state = state * player
        policy, value = super().__call__(state)
        return policy.tolist()[0], value.item() * player

    def preprocess(self, s):
        s = torch.tensor(s)
        s = s.to(device)
        s = s.view(-1, 3, 3)
        # Split into three channels - empty pieces, own pieces and enemy pieces. Will represent this with a 1
        empty_channel = (s == torch.tensor(0).to(device)).clone().float().detach()
        own_channel = (s == torch.tensor(1).to(device)).clone().float().detach()
        enemy_channel = (s == torch.tensor(-1).to(device)).clone().float().detach()
        x = torch.stack([empty_channel, own_channel, enemy_channel], 1).to(device)  # stack along channel dimension

        return x

    def forward(self, s):
        x = self.preprocess(s)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        # x = x.view(x.size(0), -1)

        policy = F.leaky_relu(self.policy_bn(self.conv_policy(x))).view(x.size(0), -1)
        policy = self.softmax(self.linear_policy(policy))

        value = F.leaky_relu(self.value_bn(self.conv_value(x))).view(x.size(0), -1)
        value = F.leaky_relu(self.fc_value(value))
        value = torch.tanh(self.linear_output(value))

        return policy, value
