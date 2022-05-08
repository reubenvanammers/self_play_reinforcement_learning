import torch
from torch import nn
from torch.functional import F
from torch.nn import functional

from games.connect4.modules import device
from games.general.base_env import BaseEnv, TwoDEnv
from rl_utils.weights import init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResidualTower(nn.Module):
    def __init__(self, width=7, height=6, action_size=7, num_blocks=15, default_kernel_size=3):
        super(ResidualTower, self).__init__()
        self.inplanes = 128
        self.width = width
        self.height = height

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=default_kernel_size, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = self._make_layer(BasicBlock, 128, num_blocks)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(width))), 1, 1, 0)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(height))), 1, 1, 0)
        linear_input_size = convw * convh

        # Policy Head
        self.conv_policy = nn.Conv2d(self.inplanes, 32, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_dropout = nn.Dropout(p=0.5)
        self.linear_policy = nn.Linear(linear_input_size * 32, action_size)
        self.softmax = nn.Softmax()

        # Value head
        self.conv_value = nn.Conv2d(self.inplanes, 32, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_dropout = nn.Dropout(p=0.5)
        self.fc_value = nn.Linear(linear_input_size * 32, 256)
        self.linear_output = nn.Linear(256, 1)

        self.apply(init_weights)

    @staticmethod
    def from_env(env: TwoDEnv, num_blocks=15):
        return ResidualTower(env.width, env.height, env.num_actions(), num_blocks)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = preprocess(x, self.width, self.height)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_blocks(x)

        policy = F.relu(self.policy_bn(self.conv_policy(x))).view(x.size(0), -1)
        policy = self.policy_dropout(policy)
        # policy = F.dropout(policy, p=0.3, training=True)  # change training method
        policy = self.softmax(self.linear_policy(policy))

        value = F.relu(self.value_bn(self.conv_value(x))).view(x.size(0), -1)
        value = self.value_dropout(value)
        # value = F.dropout(value, p=0.3, training=True)  # change training method
        value = F.relu(self.fc_value(value))
        value = torch.tanh(self.linear_output(value))

        return policy, value

    def __call__(self, state, player=1):
        state = state * player
        policy, value = super().__call__(state)
        return policy.tolist()[0], value.item() * player


def preprocess(s, width=7, height=6):
    s = torch.tensor(s)
    s = s.to(device)
    s = s.view(-1, width, height)
    # Split into three channels - empty pieces, own pieces and enemy pieces. Will represent this with a 1
    empty_channel = (s == torch.tensor(0).to(device)).clone().float().detach()
    own_channel = (s == torch.tensor(1).to(device)).clone().float().detach()
    enemy_channel = (s == torch.tensor(-1).to(device)).clone().float().detach()
    x = torch.stack([empty_channel, own_channel, enemy_channel], 1).to(device)  # stack along channel dimension

    return x


def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
    return (size + padding * 2 - (kernel_size - 1) - 1) // stride + 1
