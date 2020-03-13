from collections import namedtuple
import copy
import numpy as np
import torch
from anytree import NodeMixin
from torch import nn
from torch.functional import F
import random

from rl_utils.memory import Memory
from rl_utils.weights import init_weights

Move = namedtuple(
    "Move",
    ("state", "action", "predicted_val", "actual_val", "network_probs", "tree_probs"),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCNode(NodeMixin):
    # Represents an action of a Monte Carlo Search Tree

    def __init__(self, state=None, n=0, w=0, p=0, parent=None, cpuct=2, player=1):
        self.state = state
        self.n = n
        self.w = w
        self.p = p
        self.parent = parent
        self.player = -1 * self.parent.player if self.parent else 1

        self.cpuct = cpuct

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
        # action_probs = list(action_probs)  # TODO make everything torch native if possible
        children_list = []
        for action_prob in action_probs:  # .tolist()[0]:
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
    def __init__(self, evaluator, env_gen, iterations=100, temperature_cutoff=5, batch_size=64):
        self.iterations = iterations
        self.evaluator = evaluator
        self.env_gen = env_gen
        self.env = env_gen()
        self.reset()

        self.temp_memory = []
        self.memory = Memory(5000)

        self.temperature_cutoff = temperature_cutoff
        self.actions = self.env.action_space.n

        self.optim = torch.optim.SGD(
            self.evaluator.parameters(), weight_decay=0.0001,
            momentum=0.9, lr=0.0001
        )
        self.batch_size = batch_size

    def reset(self):
        base_state = self.env.reset()
        self.root_node = MCNode(state=base_state)
        probs, v = self.evaluator(base_state)
        self.root_node.create_children(probs)
        self.moves_played = 0

        return base_state

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
        self.root_node.children = [self.root_node.children[action]]
        self.root_node = self.root_node.children[0]

    def update(self, s, a, r, done, next_s):
        if done:
            for experience in self.temp_memory:
                experience.actual_val = r
                self.memory.add(experience)
            self.temp_memory = []
        # TODO atm start of with running update, then maybe move to async model like in paper
        if self.ready:
            self.update_from_memory()

    def update_from_memory(self):
        batch = self.memory.sample(self.batch_size)
        batch_t = Move(*zip(*batch))  # transposed batch
        s, a, predict_val, actual_val, net_probs, tree_probs = batch_t
        s_batch = torch.stack(s)
        a_batch = torch.cat(a)
        predict_val_batch = torch.cat(predict_val)
        actual_val_batch = torch.cat(actual_val)
        net_probs_batch = torch.stack(net_probs)
        tree_probs_batch = torch.stack(tree_probs)

        value_loss = F.mse_loss(predict_val_batch, actual_val_batch)
        prob_loss = F.cross_entropy(net_probs_batch, tree_probs_batch)

        loss = value_loss + prob_loss
        self.optim.zero_grad()
        loss.backward()
        for param in self.evaluator.parameters():  # see if this ends up doing anything - should just be relu
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def play(self, temp=0.01):
        play_probs = [child.play_prob(temp) / self.iterations for child in self.root_node.children]
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
        env.set_state(copy.copy(parent_node.state))
        s, r, done, _ = env.step(action, player=player)
        if done:
            v = r
            child_node = parent_node.children[action]
        else:
            probs, v = self.evaluator(s, parent_node.player)
            # TODO check if this works correctly with player - might have to swap
            child_node = parent_node.children[action]
            child_node.create_children(probs)
        child_node.state = s
        return child_node, v

    def search(self):
        for i in range(self.iterations):
            node = self.root_node
            while True:
                select_probs = [child.select_prob for child in node.children]
                try:
                    action = np.argmax(select_probs + 0.000001 * np.random.rand(self.actions))
                except ValueError:
                    action = np.random.choice(np.arange(self.actions))
                if node.children[action].is_leaf:
                    node, v = self.expand_node(node, action, node.player)
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
    def __init__(self, width=3, height=3, action_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 128, kernel_size=3, stride=1, padding=1, bias=True
        )  # Deal with padding?
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + padding * 2 - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(width))), 1, 1, 0)
        convh = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(conv2d_size_out(height))), 1, 1, 0)
        linear_input_size = convw * convh

        # Policy Head
        self.conv_policy = nn.Conv2d(64, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.linear_policy = nn.Linear(linear_input_size * 2, action_size)
        self.softmax = nn.Softmax()

        # Value head
        self.conv_value = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
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
        x = torch.stack(
            [empty_channel, own_channel, enemy_channel], 1
        )  # stack along channel dimension

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
