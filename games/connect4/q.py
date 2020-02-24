import random
import time
from collections import deque

import gym
import numpy as np
import torch
from torch import nn
from torch.functional import F
from collections import namedtuple

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


class Q(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, s):
        if not isinstance(s, torch.Tensor):
            # s = torch.scalar_tensor(s).long()
            s = torch.from_numpy(s).long()
        # if not isinstance(a, torch.Tensor):
        #     a = torch.scalar_tensor(a).long()
        return super().__call__(s)

    def v(self, s, a):
        a = a.view(-1, 1)
        return self(s).gather(1, a)

    def update(self, s, a, r, done, s_next):
        s = torch.tensor(s)
        a = torch.tensor(a)
        r = torch.tensor(r)
        done = torch.tensor(done)
        s_next = torch.tensor(s_next)
        if len(self.memory) < self.batch_size:
            self.memory.add(Transition(s, a, r, done, s_next))
            return

        # Using batch memory
        if self.memory:
            self.memory.add(Transition(s, a, r, done, s_next))
            batch = self.memory.sample(self.batch_size)
            batch_t = Transition(*zip(*batch))  # transposed batch

        # Get expected Q values
        s_batch, a_batch, r_batch, done_batch, s_next_batch = batch_t
        s_batch = torch.stack(s_batch)
        a_batch = torch.stack(a_batch)
        r_batch = torch.stack(r_batch).view(-1, 1)
        s_next_batch = torch.stack(s_next_batch)
        done_batch = torch.stack(done_batch).view(-1, 1)
        q = self.v(s_batch, a_batch)

        # Get Actual Q values
        q_next = (
            self(s_next_batch).max(1)[0].view(-1, 1).detach()
        )  # check how detach works (might be dodgy???) #max results in values and
        q_next_actual = (~done_batch) * q_next  # Removes elements that are done
        q_target = r_batch + self.gamma * q_next_actual
        loss = F.smooth_l1_loss(q, q_target)

        self.optim.zero_grad()
        loss.backward()
        for param in self.parameters():  # see if this ends up doing anything - should just be relu
            param.grad.data.clamp_(-1, 1)
        self.optim.step()


class QLinear(Q):

    def __init__(self, env, lr=0.025, gamma=1, momentum=0, buffer_size=50000, batch_size=8, weight_decay=0.5):
        super().__init__()

        self.gamma = gamma
        self.env = env
        self.state_size = self.env.width * self.env.height
        self.linear = nn.Linear(self.state_size, self.env.action_space.n)
        self.linear.weight.data.fill_(0)
        self.optim = torch.optim.SGD(self.parameters(), momentum=momentum, lr=lr, weight_decay=weight_decay)
        if buffer_size:
            self.memory = Memory(buffer_size)
        self.batch_size = batch_size

    def forward(self, s):  # At the moment just use a linear network - dont expect it to be good
        s = s.view(-1, self.state_size).float()

        return self.linear(s)


class QConv(Q):

    def __init__(self, env, lr=0.025, gamma=1, momentum=0, buffer_size=50000, batch_size=8, weight_decay=0):
        super().__init__()

        self.gamma = gamma
        self.env = env
        self.state_size = self.env.width * self.env.height
        # self.linear = nn.Linear(self.state_size, self.env.action_space.n)
        # self.linear.weight.data.fill_(0.5)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)  # Deal with padding?
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=1, padding=2):
            return (size + padding * 2 - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.env.width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.env.height)))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, self.env.action_space.n)

        self.optim = torch.optim.SGD(self.parameters(), momentum=momentum, lr=lr, weight_decay=weight_decay)
        if buffer_size:
            self.memory = Memory(buffer_size)
        self.batch_size = batch_size

    def forward(self, s):
        s = s.view(-1, 7, 6)
        # Split into three channels - empty pieces, own pieces and enemy pieces. Will represent this with a 1
        empty_channel = torch.tensor(s == 0, dtype=torch.float).detach()
        own_channel = torch.tensor(s == 1, dtype=torch.float).detach()
        enemy_channel = torch.tensor(s == -1, dtype=torch.float).detach()
        x = torch.stack([empty_channel, own_channel, enemy_channel], 1)  # stack along channel dimension

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
