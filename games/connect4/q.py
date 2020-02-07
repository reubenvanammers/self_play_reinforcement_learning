import random
import time
from collections import deque

import gym
import numpy as np
import torch
from torch import nn
from torch.functional import F


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        return [self.buffer[i] for i in index]


class EpsilonGreedy:
    def __init__(self, q, epsilon):
        self.q = q
        self.epsilon = epsilon

    def __call__(self, s):
        if np.random.rand() < self.epsilon:
            a = self.q.env.action_space.sample()
        else:
            a = max(range(self.q.env.action_space.n), key=(lambda a_: self.q(s, a_).item()))
        return a


class QLinear(nn.Module):
    def __call__(self, s, a):
        if not isinstance(s, torch.Tensor):
            # s = torch.scalar_tensor(s).long()
            s = torch.from_numpy(s).long()
        if not isinstance(a, torch.Tensor):
            a = torch.scalar_tensor(a).long()
        return super().__call__(s, a)

    def __init__(self, env, lr=0.025, gamma=1, momentum=0):
        super().__init__()

        self.gamma = gamma
        self.env = env
        self.linear = nn.Linear(self.env.width * self.env.width * self.env.height, 1)
        self.linear.weight.data.fill_(0.5)
        self.optim = torch.optim.SGD(self.parameters(), momentum=momentum, lr=lr)

    def forward(self, s, a):  # At the moment just use a linear network - dont expect it to be good
        a = nn.functional.one_hot(a, self.env.width)
        # s = nn.functional.one_hot(s, output_size)
        s = s.view(-1)
        x = torch.functional.einsum("i,j->ij", s, a).float().view(-1)  # Cartesian product

        return self.linear(x)

    def update(self, s, a, r, done, s_next):
        # prediction
        q = self(s, a)

        # actual
        q_next = max(self(s_next, a_) for a_ in range(self.env.action_space.n))  # Check!
        q_target = r + (1 - done) * self.gamma * q_next
        #         q_target = torch.scalar_tensor(q_target).unsqueeze(0)
        loss = F.smooth_l1_loss(q, q_target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
