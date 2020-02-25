import random
import time
from collections import deque

import gym
import numpy as np
import torch
from torch import nn
from torch.functional import F
from collections import namedtuple

from rl_utils.sum_tree import WeightedMemory
from rl_utils.losses import weighted_smooth_l1_loss

Transition = namedtuple("Transition", ("state", "action", "reward", "done", "next_state"))


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        return [self.buffer[i] for i in index]

    def reset(self):
        self.buffer = deque(maxlen=self.max_size)


class EpsilonGreedy:
    def __init__(self, q, epsilon):
        self.q = q
        self.epsilon = epsilon

    def __call__(self, s):
        if np.random.rand() < self.epsilon:
            possible_moves = [i for i, move in enumerate(self.q.env.valid_moves()) if move]
            a = random.choice(possible_moves)
        else:
            # a = max(range(self.q.env.action_space.n), key=(lambda a_: self.q(s, a_).item()))
            weights = self.q(s).detach().numpy()
            mask = (-1000000000 * ~np.array(
                self.q.env.valid_moves())) + 1  # just a really big negative number? is quite hacky
            a = np.argmax(weights + mask)
        return a


class Q:
    def __init__(self, mem_type='sumtree', buffer_size=20000, batch_size=16, *args, **kwargs):
        if mem_type == 'sumtree':
            self.memory = WeightedMemory(buffer_size)
        else:
            self.memory = Memory(buffer_size)
        self.batch_size = batch_size

    def __call__(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.from_numpy(s).long()
        # return super().__call__(s)
        return self.policy_net(s)

    def state_action_value(self, s, a):
        a = a.view(-1, 1)
        return self.policy_net(s).gather(1, a)

    def update(self, s, a, r, done, s_next):
        s = torch.tensor(s)
        a = torch.tensor(a)
        r = torch.tensor(r)
        done = torch.tensor(done)
        s_next = torch.tensor(s_next)
        if len(self.memory) < self.memory.max_size:
            self.memory.add(Transition(s, a, r, done, s_next))
            return

        # Using batch memory
        self.memory.add(Transition(s, a, r, done, s_next))
        if isinstance(self.memory, WeightedMemory):
            tree_idx, batch, sample_weights = self.memory.sample(self.batch_size)
            sample_weights = torch.tensor(sample_weights)
        else:
            batch = self.memory.sample(self.batch_size)
        batch_t = Transition(*zip(*batch))  # transposed batch

        # Get expected Q values
        s_batch, a_batch, r_batch, done_batch, s_next_batch = batch_t
        s_batch = torch.stack(s_batch)
        a_batch = torch.stack(a_batch)
        r_batch = torch.stack(r_batch).view(-1, 1)
        s_next_batch = torch.stack(s_next_batch)
        done_batch = torch.stack(done_batch).view(-1, 1)
        q = self.state_action_value(s_batch, a_batch)

        # Get Actual Q values

        double_actions = self.policy_net(s_next_batch).max(1)[1].detach()  # used for double q learning
        q_next = self.state_action_value(s_next_batch, double_actions)

        # q_next = (
        #     self.target_net(s_next_batch).max(1)[0].view(-1, 1).detach()
        # )  # check how detach works (might be dodgy???) #max results in values and
        q_next_actual = (~done_batch) * q_next  # Removes elements that are done
        q_target = r_batch + self.gamma * q_next_actual
        ###TEST if clamping works or is even good practise
        q_target = q_target.clamp(-1, 1)
        ###/TEST

        if isinstance(self.memory, WeightedMemory):
            absolute_loss = torch.abs(q - q_target).detach().numpy()
            loss = weighted_smooth_l1_loss(q,
                                           q_target,
                                           sample_weights)  # TODO fix potential non-linearities using huber loss
            self.memory.batch_update(tree_idx, absolute_loss)

        else:
            loss = F.smooth_l1_loss(q, q_target)

        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # see if this ends up doing anything - should just be relu
            param.grad.data.clamp_(-1, 1)
        self.optim.step()


class QLinear(Q):

    def __init__(self, env, lr=0.025, gamma=0.99, momentum=0, weight_decay=0.5, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)  # gamma is slightly less than 1 to promote faster games

        self.gamma = gamma
        self.env = env
        self.state_size = self.env.width * self.env.height
        self.linear = nn.Linear(self.state_size, self.env.action_space.n)
        self.linear.weight.data.fill_(0)
        self.optim = torch.optim.RMSprop(self.parameters(), momentum=momentum, lr=lr, weight_decay=weight_decay)

    def forward(self, s):  # At the moment just use a linear network - dont expect it to be good
        s = s.view(-1, self.state_size).float()

        return self.linear(s)


class ConvNet(nn.Module):

    def __init__(self, width, height, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)  # Deal with padding?
        self.bn1 = nn.BatchNorm2d(16)
        # self.drop1 = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        # self.drop2 = nn.Dropout2d(p=0.1)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(32)

        # self.drop3 = nn.Dropout2d(p=0.1)

        # self.conv1.weight.data.fill_(0.01)
        # self.conv2.weight.data.fill_(0.01)
        # self.conv3.weight.data.fill_(0.01)
        # self.conv1.bias.data.fill_(0.0)
        # self.conv2.bias.data.fill_(0.0)
        # self.conv3.bias.data.fill_(0.0)
        # self.bn1.weight.data.fill_(0.0)
        # self.bn2.weight.data.fill_(0.0)
        # self.bn3.weight.data.fill_(0.0)

        def conv2d_size_out(size, kernel_size=5, stride=1, padding=2):
            return (size + padding * 2 - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = convw * convh * 32

        self.value_fc = nn.Linear(linear_input_size, 512)
        self.value = nn.Linear(512, 1)

        self.advantage_fc = nn.Linear(linear_input_size, 512)
        self.advantage = nn.Linear(512, action_size)

        # self.head = nn.Linear(linear_input_size, action_size)

    def forward(self, s):
        s = s.view(-1, 7, 6)
        # Split into three channels - empty pieces, own pieces and enemy pieces. Will represent this with a 1
        empty_channel = (s == 0).clone().float().detach()
        own_channel = (s == 1).clone().float().detach()
        enemy_channel = (s == -1).clone().float().detach()
        x = torch.stack([empty_channel, own_channel, enemy_channel], 1)  # stack along channel dimension
        # print(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        # print(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        # print(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        # print(x)

        value = self.value(self.value_fc(x.view(x.size(0), -1)))
        advantage = self.advantage(self.advantage_fc(x.view(x.size(0), -1)))

        output = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
        return output
        # return self.head(x.view(x.size(0), -1))


class QConv(Q):

    def __init__(self, env, lr=0.025, gamma=0.99, momentum=0, weight_decay=0, *args,
                 **kwargs):
        # gamma is slightly less than 1 to promote faster games
        super().__init__(*args, **kwargs)  # gamma is slightly less than 1 to promote faster games

        self.gamma = gamma
        self.env = env
        self.state_size = self.env.width * self.env.height
        self.policy_net = ConvNet(self.env.width, self.env.height, self.env.action_space.n)
        self.target_net = ConvNet(self.env.width, self.env.height, self.env.action_space.n)
        # self.linear = nn.Linear(self.state_size, self.env.action_space.n)
        # self.linear.weight.data.fill_(0.5)

        self.optim = torch.optim.RMSprop(self.policy_net.parameters(),
                                         weight_decay=weight_decay)  # , momentum=momentum, lr=lr, weight_decay=weight_decay)