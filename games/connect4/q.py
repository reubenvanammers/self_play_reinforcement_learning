import random
import time
from collections import deque

import gym
import numpy as np
import torch
from torch import nn
from torch.functional import F
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

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
            # a = max(range(self.q.env.action_space.n), key=(lambda a_: self.q(s, a_).item()))
            a = np.argmax(self.q(s).detach().numpy())
        return a


class QLinear(nn.Module):
    def __call__(self, s):
        if not isinstance(s, torch.Tensor):
            # s = torch.scalar_tensor(s).long()
            s = torch.from_numpy(s).long()
        # if not isinstance(a, torch.Tensor):
        #     a = torch.scalar_tensor(a).long()
        return super().__call__(s)

    def __init__(
        self, env, lr=0.025, gamma=1, momentum=0, buffer_size=50000, batch_size=16
    ):
        super().__init__()

        self.gamma = gamma
        self.env = env
        self.state_size = self.env.width * self.env.height
        self.linear = nn.Linear(self.state_size, self.env.action_space.n)
        self.linear.weight.data.fill_(0.5)
        self.optim = torch.optim.SGD(self.parameters(), momentum=momentum, lr=lr)
        if buffer_size:
            self.memory = Memory(buffer_size)
        self.batch_size = batch_size

    def forward(
        self, s
    ):  # At the moment just use a linear network - dont expect it to be good
        # a = nn.functional.one_hot(a, self.env.width)
        # # s = nn.functional.one_hot(s, output_size)
        s = s.view(-1, self.state_size).float()
        # s = s.unsqueeze(0).float()
        # x = torch.functional.einsum("i,j->ij", s, a).float().view(-1)  # Cartesian product

        return self.linear(s)

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

        # prediction
        if self.memory:
            self.memory.add(Transition(s, a, r, done, s_next))
            batch = self.memory.sample(self.batch_size)
            batch_t = Transition(*zip(*batch))  # transposed batch
        # q = self(s, a)

        # prediction
        s_batch, a_batch, r_batch, done_batch, s_next_batch = batch_t
        s_batch = torch.stack(s_batch)
        a_batch = torch.stack(a_batch)
        r_batch = torch.stack(r_batch)
        s_next_batch = torch.stack(s_next_batch)
        done_batch = torch.stack(done_batch)
        q = self.v(s_batch, a_batch)
        # seperate state,actions

        # actual
        q_next = (
            self(s_next_batch).max(1)[0].detach()
        )  # check how detach works (might be dodgy???) #max results in values and
        # q_next = max(self(s_next, a_) for a_ in range(self.env.action_space.n))  # Check!
        q_next_actual = (~done_batch) * q_next  # Removes elements that are done
        q_target = r_batch + self.gamma * q_next_actual
        # q_target = r + (1 - done) * self.gamma * q_next
        #         q_target = torch.scalar_tensor(q_target).unsqueeze(0)
        loss = F.smooth_l1_loss(q, q_target)

        self.optim.zero_grad()
        loss.backward()
        for (
            param
        ) in (
            self.linear.parameters()
        ):  # see if this ends up doing anything - should just be relu
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
