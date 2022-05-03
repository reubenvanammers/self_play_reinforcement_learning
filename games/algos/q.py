import copy
import random
from collections import namedtuple

import numpy as np
import torch
from torch.functional import F

import games.general.modules
from games.general.base_model import BaseModel, BasePlayer, Policy
from rl_utils.losses import weighted_smooth_l1_loss
from rl_utils.memory import Memory
from rl_utils.sum_tree import WeightedMemory
from rl_utils.weights import init_weights

Transition = namedtuple("Transition", ("state", "action", "reward", "done", "next_state"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpsilonGreedy(Policy):
    # TODO Fix this
    MEM_TYPE = "sumtree"

    def __init__(
        self,
        evaluator,
        epsilon,
        env_gen,
        optim=None,
        memory_queue=None,
        memory_size=20000,
        mem_type=None,
        batch_size=64,
        gamma=0.99,
    ):
        self.epsilon = epsilon
        self._epsilon = epsilon  # Backup for evaluatoin
        self.optim = optim
        self.env = env_gen()
        self.memory_queue = memory_queue
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = copy.deepcopy(evaluator)
        self.target_net = copy.deepcopy(evaluator)

        super().__init__(memory_size, memory_queue)

    def create_memory(self, memory_size):
        if self.MEM_TYPE == "sumtree":
            self.memory = WeightedMemory(memory_size)
        else:
            self.memory = Memory(memory_size)

    def __call__(self, s):
        return self._epsilon_greedy(s)

    def q(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.from_numpy(s).long()
        # s = self.policy_net.preprocess(s)
        return self.policy_net(s)  # Only get predict policiies

    def _epsilon_greedy(self, s):
        if np.random.rand() < self.epsilon:
            possible_moves = [i for i, move in enumerate(self.env.valid_moves()) if move]
            a = random.choice(possible_moves)
        else:
            weights = self.q(s).detach().cpu().numpy()  # TODO maybe do this with tensors
            mask = (
                -1000000000 * ~np.array(self.env.valid_moves())
            ) + 1  # just a really big negative number? is quite hacky
            a = np.argmax(weights + mask)
        return a

    def load_state_dict(self, state_dict, target=False):
        self.policy_net.load_state_dict(state_dict)
        if target:
            self.target_net.load_state_dict(state_dict)

    def update(self, s, a, r, done, next_s):
        self.push_to_queue(s, a, r, done, next_s)
        self.pull_from_queue()
        if self.ready:
            self.update_from_memory()

    def push_to_queue(self, s, a, r, done, next_s):
        s = torch.tensor(s, device=device)
        a = torch.tensor(a, device=device)
        r = torch.tensor(r, device=device)
        done = torch.tensor(done, device=device)
        next_s = torch.tensor(next_s, device=device)
        self.memory_queue.put(Transition(s, a, r, done, next_s))

    def update_from_memory(self):
        if isinstance(self.memory, WeightedMemory):
            tree_idx, batch, sample_weights = self.memory.sample(self.batch_size)
            sample_weights = torch.tensor(sample_weights, device=device)
        else:
            batch = self.memory.sample(self.batch_size)
        batch_t = Transition(*zip(*batch))  # transposed batch
        s_batch, a_batch, r_batch, done_batch, s_next_batch = batch_t
        s_batch = torch.cat(s_batch)
        a_batch = torch.stack(a_batch)
        r_batch = torch.stack(r_batch).view(-1, 1)
        s_next_batch = torch.cat(s_next_batch)
        done_batch = torch.stack(done_batch).view(-1, 1)
        q = self._state_action_value(s_batch, a_batch)

        # Get Actual Q values

        double_actions = self.policy_net(s_next_batch).max(1)[1].detach()  # used for double q learning
        q_next = self._state_action_value(s_next_batch, double_actions)

        q_next_actual = (~done_batch) * q_next  # Removes elements thx`at are done
        q_target = r_batch + self.gamma * q_next_actual
        q_target = q_target.clamp(-1, 1)

        if isinstance(self.memory, WeightedMemory):
            absolute_loss = torch.abs(q - q_target).detach().cpu().numpy()
            loss = weighted_smooth_l1_loss(
                q, q_target, sample_weights
            )  # TODO fix potential non-linearities using huber loss
            self.memory.batch_update(tree_idx, absolute_loss)

        else:
            loss = F.smooth_l1_loss(q, q_target)

        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # see if this ends up doing anything - should just be relu
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    # determines when a neural net has enough data to train
    @property
    def ready(self):
        return len(self.memory) >= self.memory.max_size

    def state_dict(self):
        return self.policy_net.state_dict()

    def update_target_net(self):
        self.target_net.load_state_dict(self.state_dict())

    def train(self, train_state=True):
        return self.policy_net.train(train_state)

    def reset(self, *args, **kwargs):
        self.env.reset()

    def _state_action_value(self, s, a):
        a = a.view(-1, 1)
        return self.policy_net(s).gather(1, a)

    def evaluate(self, evaluate_state=False):
        # like train - sets evaluate state
        if evaluate_state:
            # self._epsilon = self.epsilon
            self.epsilon = 0
        else:
            self.epsilon = self._epsilon
        # self.evaluating = evaluate_state

    def play_action(self, action, player):
        self.env.step(action, player)
        # pass  # does nothign atm - mostly for the mcts


class Q:
    def __init__(
        self,
        env,
        evaluator,
        lr=0.01,
        gamma=0.99,
        momentum=0.9,
        weight_decay=0.01,
        mem_type="sumtree",
        buffer_size=20000,
        batch_size=16,
        *args,
        **kwargs
    ):

        self.gamma = gamma
        self.env = env
        self.state_size = self.env.width * self.env.height
        self.policy_net = copy.deepcopy(evaluator)
        # ConvNetConnect4(self.env.width, self.env.height, self.env.action_space.n).to(device)
        self.target_net = copy.deepcopy(evaluator)
        # ConvNetConnect4(self.env.width, self.env.height, self.env.action_space.n).to(device)

        self.policy_net.apply(init_weights)
        self.target_net.apply(init_weights)

        self.optim = torch.optim.SGD(
            self.policy_net.parameters(),
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
        )

        if mem_type == "sumtree":
            self.memory = WeightedMemory(buffer_size)
        else:
            self.memory = Memory(buffer_size)
        self.batch_size = batch_size

    def __call__(self, s, player=None):  # TODO use player variable
        if not isinstance(s, torch.Tensor):
            s = torch.from_numpy(s).long()
        s = games.general.modules.preprocess(s)
        return self.policy_net(s)

    def state_action_value(self, s, a):
        a = a.view(-1, 1)
        return self.policy_net(s).gather(1, a)

    def update(self, s, a, r, done, s_next):
        s = torch.tensor(s, device=device)
        # s = self.policy_net.preprocess(s)
        a = torch.tensor(a, device=device)
        r = torch.tensor(r, device=device)
        done = torch.tensor(done, device=device)
        s_next = torch.tensor(s_next, device=device)
        # s_next = self.policy_net.preprocess(s_next)

        if not self.ready:
            self.memory.add(Transition(s, a, r, done, s_next))
            return

        # Using batch memory
        self.memory.add(Transition(s, a, r, done, s_next))
        if isinstance(self.memory, WeightedMemory):
            tree_idx, batch, sample_weights = self.memory.sample(self.batch_size)
            sample_weights = torch.tensor(sample_weights, device=device)
        else:
            batch = self.memory.sample(self.batch_size)
        batch_t = Transition(*zip(*batch))  # transposed batch

        # Get expected Q values
        s_batch, a_batch, r_batch, done_batch, s_next_batch = batch_t
        s_batch = torch.cat(s_batch)
        a_batch = torch.stack(a_batch)
        r_batch = torch.stack(r_batch).view(-1, 1)
        s_next_batch = torch.cat(s_next_batch)
        done_batch = torch.stack(done_batch).view(-1, 1)
        q = self.state_action_value(s_batch, a_batch)

        # Get Actual Q values

        double_actions = self.policy_net(s_next_batch).max(1)[1].detach()  # used for double q learning
        q_next = self.state_action_value(s_next_batch, double_actions)

        q_next_actual = (~done_batch) * q_next  # Removes elements thx`at are done
        q_target = r_batch + self.gamma * q_next_actual
        ###TEST if clamping works or is even good practise
        q_target = q_target.clamp(-1, 1)
        ###/TEST

        if isinstance(self.memory, WeightedMemory):
            absolute_loss = torch.abs(q - q_target).detach().cpu().numpy()
            loss = weighted_smooth_l1_loss(
                q, q_target, sample_weights
            )  # TODO fix potential non-linearities using huber loss
            self.memory.batch_update(tree_idx, absolute_loss)

        else:
            loss = F.smooth_l1_loss(q, q_target)

        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # see if this ends up doing anything - should just be relu
            param.grad.data.clamp_(-1, 1)
        self.optim.step()


class QConvConnect4(Q):
    def __init__(self, env, lr=0.01, gamma=0.99, momentum=0.9, weight_decay=0.01, *args, **kwargs):
        # gamma is slightly less than 1 to promote faster games
        super().__init__(*args, **kwargs)
