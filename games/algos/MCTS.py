from collections import namedtuple

import numpy as np
import torch
from anytree import NodeMixin
from torch import nn
from torch.functional import F
import random

from rl_utils.memory import Memory

Move = namedtuple(
    "Move",
    ("state", "action", "predicted_val", "actual_val", "network_probs", "tree_probs"),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCNode(NodeMixin):
    # Represents an action of a Monte Carlo Search Tree

    def __init__(self, state=None, n=0, w=0, p=0, parent=None, player=1):
        self.state = state
        self.n = n
        self.w = w
        self.p = p
        self.parent = parent
        self.player = -1 * self.parent.player if self.parent else 1

        self.cpuct = 1

        self.v = None

    @property
    def q(self):
        return self.w / self.n if self.n else 0

    @property
    def u(self):
        return self.cpuct * self.p * np.sqrt(self.parent.n) / (1 + self.n)

    @property
    def select_prob(self):
        return self.q + self.u

    def backup(self, v):
        self.w += v
        self.n += 1
        if self.parent:
            self.parent.backup(v)

    def play_prob(self, temp):
        return np.power(self.n, 1 / temp)

    def create_children(self, action_probs):
        children_list = []
        for action_prob in action_probs:
            children_list.append(MCNode(p=action_prob, parent=self))
        self.children = children_list

    def _post_detach_children(self, children):
        for child in children:
            if child.children:
                child.children = []
            del child


# TODO deal with opposite player choosing moves
# TODO Start off with opponent using their own policy (eg random) and then move to MCTS as well
class MCTreeSearch:
    def __init__(self, evaluator, env_gen, iterations=100, actions=7, temperature_cutoff=5):
        self.iterations = iterations
        self.evaluator = evaluator
        self.env_gen = env_gen
        self.env = env_gen()
        base_state = self.env.reset()
        self.root_node = MCNode(state=base_state)

        self.temp_memory = []
        self.memory = Memory(5000)

        self.temperature_cutoff = temperature_cutoff
        self.actions = actions
        self.moves_played = 0

        self.optim = torch.optim.SGD(
            self.evaluator.parameters(), weight_decay=0.0001,
            momentum=0.9, lr=0.0001
        )

    ## Ignores the inputted state for the moment. Produces the correct action, and changes the root node appropriately
    # TODO might want to do a check on the state to make sure it is consistent
    def __call__(self, s):
        move = self.search_and_play()
        return move
        # if np.random.rand() < self.epsilon:
        #     possible_moves = [i for i, move in enumerate(self.q.env.valid_moves()) if move]
        #     a = random.choice(possible_moves)
        # else:
        #     # a = max(range(self.q.env.action_space.n), key=(lambda a_: self.q(s, a_).item()))
        #     weights = self.q(s).detach().cpu().numpy()  # TODO maybe do this with tensors
        #     mask = (
        #                    -1000000000 * ~np.array(self.q.env.valid_moves())
        #            ) + 1  # just a really big negative number? is quite hacky
        #     a = np.argmax(weights + mask)
        # return a

    def search_and_play(self):
        temperature = 1 if self.moves_played < self.temperature_cutoff else 0
        self.search()
        move = self.play(temperature)
        return move

    def opponent_action(self, action):
        self.prune(action)

    def prune(self, action):
        # Choose action - and remove all other elements of tree
        self.root_node.children = self.root_node.children[action]
        self.root_node = self.root_node.children[0]

    def update(self, s, a, r, done, next_s):
        if done:
            for experience in self.temp_memory:
                experience.actual_val = r
                self.memory.add(experience)
            self.temp_memory = []
        # TODO atm start of with running update, then maybe move to async model like in paper

    def play(self, temp=0.01):
        play_probs = [child.play_prob(temp) for child in self.root_node.children]
        move_probs = [child.p for child in self.root_node.children]

        action = np.random.choice(self.actions, p=play_probs)
        self.prune(action)

        self.moves_played += 1

        self.temp_memory.append(
            Move(
                self.root_node.state,
                action,
                self.root_node.v,
                None,
                move_probs,
                play_probs,
            )
        )
        return action

    def expand_node(self, parent_node, action, player=1):
        env = self.env_gen()
        env.set_state(parent_node.state)
        s, r, done, _ = env.step(action, player=player)
        if done:
            v = r
            child_node = parent_node.children[action]
        else:
            probs, v = self.evaluator(s, parent_node.player)
            # TODO check if this works correctly with player - might have to swap
            child_node = parent_node.children[action]
            child_node.create_children(probs)
        return child_node, v

    def search(self):
        node = self.root_node
        for i in range(self.iterations):
            while True:
                select_probs = [child.select_prob for child in node.children]
                action = np.random.choice(self.actions, p=select_probs)
                if node.is_leaf:
                    node, v = self.expand_node(node, action)
                    node.backup(v)
                    node.v = v
                    break
                else:
                    node = node.children[action]

    def load_state_dict(self, state_dict, target=False):
        self.evaluator.load_state_dict(state_dict)

    # determines when a neural net has enough data to train
    @property
    def ready(self):
        # Hard code value for the moment
        return len(self.memory) >= 1000

    def state_dict(self):
        return self.evaluator.state_dict()

    def update_target_net(self):
        # No target net so pass
        pass

    def train(self, train_state):
        return self.evaluator.train(train_state)


class ConvNetTicTacToe(nn.Module):
    def __init__(self, width, height, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(
            4, 128, kernel_size=3, stride=1, padding=1, bias=True
        )  # Deal with padding?
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + padding * 2 - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 1, 1, 0)))
        )
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 1, 1, 0)))
        )
        linear_input_size = convw * convh * 64

        # Policy Head
        self.conv_policy = nn.Conv2d(64, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.linear_policy = nn.Linear(linear_input_size, action_size)

        # Value head
        self.conv_value = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.fc_value = nn.Linear(linear_input_size, 256)
        self.linear_output = nn.Linear(256, 1)

    def preprocess(self, s):
        s = s.to(device)
        s = s.view(-1, 3, 3)
        # Split into three channels - empty pieces, own pieces and enemy pieces. Will represent this with a 1
        empty_channel = (s == torch.tensor(0).to(device)).clone().float().detach()
        own_channel = (s == torch.tensor(1).to(device)).clone().float().detach()
        enemy_channel = (s == torch.tensor(-1).to(device)).clone().float().detach()
        x = torch.stack(
            [empty_channel, own_channel, enemy_channel, s.float().detach()], 1
        )  # stack along channel dimension

        return x

    def forward(self, s):
        x = self.preprocess(s)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        policy = F.leaky_relu(self.policy_bn(self.conv_policy(x)))
        policy = self.linear_policy(policy)

        value = F.leaky_relu(self.value_bn(self.conv_value(x)))
        value = F.leaky_relu(self.fc_value(value))
        value = torch.tanh(self.linear_output(value))

        return policy, value
