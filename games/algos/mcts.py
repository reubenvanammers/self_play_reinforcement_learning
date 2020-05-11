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

Move = namedtuple("Move", ("state", "actual_val", "tree_probs"), )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False






class MCNode(NodeMixin):
    # Represents an action of a Monte Carlo Search Tree

    def __init__(self, state=None, n=0, w=0, p=0, x=0.25, parent=None, cpuct=4, player=1, v=None, valid=True):
        self.state = state
        self.n = n
        self.w = w
        self.p = p
        self.x = x  # Dirichlet parameter - 0 means no effect, 1 means completely dirchlet

        self.noise_active = False
        self.p_noise = 0

        self.alpha = 1  # Dirichlet parameter
        self.parent = parent
        self.player = -1 * self.parent.player if self.parent else player
        self.valid = valid  # Whether an action is valid - don't play moves that you cannot
        self.cpuct = cpuct  # exploration factor

        self.active_root = False
        self.v = v

    def add_noise(self):  # Adds noise to childrens p value
        dirichlet_distribution = np.random.dirichlet([self.alpha] * len(self.children))
        for i, c in enumerate(self.children):
            c.noise_active = True
            c.p_noise = dirichlet_distribution[i]

    def remove_noise(self):
        for i, c in enumerate(self.children):
            c.noise_active = False

    @property
    def q(self):  # Attractiveness of a node from player ones pespective - average of downstream results
        return self.w / self.n if self.n else 0
        # return self.w / self.n if self.n else self.p

    @property  # effective p - p if noise isn't active, otherwise adds noise factor
    def p_eff(self):
        if self.noise_active:
            return self.p_noise * self.x + self.p * (1 - self.x)
        else:
            return self.p

    @property
    def u(self):  # Factor to encourage exploration - higher values of cpuct increase exploration
        return self.cpuct * self.p_eff * np.sqrt(self.parent.n) / (1 + self.n)

    @property
    def select_prob(self):  # TODO check if this works? might need to be ther other way round
        return -1 * self.player * self.q + self.u  # -1 is due to that this calculated from the perspective of the parent node, which has an opposite player
        # return self.player * self.q * self.u

    def backup(self, v):
        self.w += v
        self.n += 1
        if self.parent:
            self.parent.backup(v)

    def play_prob(self, temp):
        return np.power(self.n, 1 / temp)

    def create_children(self, action_probs, validities):
        # action_probs = list(action_probs)  # TODO make everything torch native if possible
        children_list = []
        for i, action_prob in enumerate(action_probs):  # .tolist()[0]:
            children_list.append(MCNode(p=action_prob, parent=self, valid=validities[i]))
            self.children = children_list

    def _post_detach_children(self, children):
        pass
        # if not self.active_root:
        #     for child in children:
        #         if child.children:
        #             child.children = []
        #         del child


# TODO deal with opposite player choosing moves
# TODO Start off with opponent using their own policy (eg random) and then move to MCTS as well
class MCTreeSearch:
    def __init__(
            self,
            evaluator,
            env_gen,
            optim=None,
            memory_queue=None,
            iterations=100,
            temperature_cutoff=5,
            batch_size=64,
            memory_size=200000,
            min_memory=20000,
            update_nn=True,
    ):
        self.iterations = iterations
        self.evaluator = evaluator.to(device)
        self.env_gen = env_gen
        self.optim = optim
        self.env = env_gen()
        self.root_node = None
        self.reset()
        self.update_nn = update_nn

        self.memory_queue = memory_queue
        self.temp_memory = []
        self.memory = Memory(memory_size)
        self.min_memory = min_memory
        self.temperature_cutoff = temperature_cutoff
        self.actions = self.env.action_space.n

        self.evaluating = False

        self.batch_size = batch_size

    def reset(self, player=1):
        base_state = self.env.reset()
        probs, v = self.evaluator(base_state)
        self.set_root(MCNode(state=base_state, v=v, player=player))
        self.root_node.create_children(probs, self.env.valid_moves())
        self.moves_played = 0
        self.temp_memory = []

        return base_state

    ## Ignores the inputted state for the moment. Produces the correct action, and changes the root node appropriately
    # TODO might want to do a check on the state to make sure it is consistent
    def __call__(self, s):  # not using player
        move = self.search_and_play()
        return move

    def search_and_play(self):
        final_temp = 1
        temperature = 1 if self.moves_played < self.temperature_cutoff else final_temp
        self.search()
        move = self.play(temperature)
        return move

    def play_action(self, action, player):
        self.set_node(action)

    def set_root(self, node):
        if self.root_node:
            self.root_node.active_root = False
        self.root_node = node
        self.root_node.active_root = True

    def prune(self, action):
        self.set_root(self.root_node.children[action])
        # Choose action - and remove all other elements of tree
        self.root_node.parent.children = [self.root_node]
        # self.root_node = self.root_node.children[0]

    def set_node(self, action):
        node = self.root_node.children[action]
        if self.root_node.children[action].n == 0:
            # TODO check if leaf and n > 0??
            # TODO might not backup??????????
            node, v = self.expand_node(self.root_node, action, self.root_node.player)
            node.backup(v)
            node.v = v
        self.set_root(node)

    def update(self, s, a, r, done, next_s):
        self.push_to_queue(done, r)
        self.pull_from_queue()
        if self.ready:
            self.update_from_memory()

    def pull_from_queue(self):
        while not self.memory_queue.empty():
            experience = self.memory_queue.get()
            self.memory.add(experience)

    def push_to_queue(self, done, r):
        if done:
            for experience in self.temp_memory:
                experience = experience._replace(actual_val=torch.tensor(r).float().to(device))
                # experience.actual_val =
                self.memory_queue.put(experience)
            self.temp_memory = []

    def loss(self, batch):
        batch_t = Move(*zip(*batch))  # transposed batch
        s, actual_val, tree_probs = batch_t
        s_batch = torch.stack(s)
        # a_batch = torch.stack(a)
        # predict_val_batch = torch.stack(predict_val)
        net_probs_batch, predict_val_batch = self.evaluator.forward(s_batch)
        predict_val_batch = predict_val_batch.view(-1)
        actual_val_batch = torch.stack(actual_val)
        # net_probs_batch = torch.stack(net_probs)
        tree_probs_batch = torch.stack(tree_probs)
        # tree_best_move = torch.argmax(tree_probs_batch, dim=1)  # TODO - fix this?

        # value_loss = F.smooth_l1_loss(predict_val_batch, actual_val_batch)

        c = MSELossFlat(floatify=True)
        value_loss = c(predict_val_batch, actual_val_batch)

        # prob_loss = F.cross_entropy(net_probs_batch, tree_best_move)
        prob_loss = - (net_probs_batch.log() * tree_probs_batch).sum() / net_probs_batch.size()[0]
        # value_loss = torch.autograd.Variable(value_loss,requires_grad=
        #                                True)
        # prob_loss = torch.autograd.Variable(prob_loss,requires_grad=
        #                                True)
        #
        loss = value_loss + prob_loss
        return loss

    def update_from_memory(self):
        batch = self.memory.sample(self.batch_size)

        loss = self.loss(batch)

        # loss = torch.autograd.Variable(loss,requires_grad=
        #                                True)
        self.optim.zero_grad()

        if APEX_AVAILABLE:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()



        # for param in self.evaluator.parameters():  # see if this ends up doing anything - should just be relu
        #     param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def play(self, temp=0.05):
        if self.evaluating:  # Might want to just make this greedy
            temp = temp / 20  # More likely to choose higher visited nodes

        play_probs = [child.play_prob(temp) for child in self.root_node.children]
        play_probs = play_probs / sum(play_probs)
        # move_probs = [child.p for child in self.root_node.children]

        action = np.random.choice(self.actions, p=play_probs)
        # self.prune(action) #Do this in the self play step

        self.moves_played += 1

        self.temp_memory.append(
            Move(
                torch.tensor(self.root_node.state).to(device),
                # torch.tensor(action).to(devi      ce),
                # torch.tensor(self.root_node.v).to(device),
                None,
                # torch.tensor(move_probs).to(device),
                torch.tensor(play_probs).float().to(device),
            )
        )
        return action

    def expand_node(self, parent_node, action, player=1):
        env = self.env_gen()
        env.set_state(copy.copy(parent_node.state))
        s, r, done, _ = env.step(action, player=player)
        r = r * player
        if done:
            v = r
            child_node = parent_node.children[action]
        else:
            probs, v = self.evaluator(s, parent_node.player)
            # TODO fix this?
            # probs = [prob if is_valid else 0 for prob, is_valid in zip(probs, env.valid_moves())]
            # TODO check if this works correctly with player - might have to swap
            child_node = parent_node.children[action]
            child_node.create_children(probs, env.valid_moves())
            assert child_node.children
        child_node.state = s
        return child_node, v

    def search(self):
        self.root_node.add_noise()  # Might want to remove this in evaluation?
        for i in range(self.iterations):
            node = self.root_node
            while True:
                select_probs = [
                    child.select_prob if child.valid else -10000000000 for child in node.children
                ]  # real big negative nuber
                action = np.argmax(select_probs + 0.000001 * np.random.rand(self.actions))
                # except ValueError:
                #     action = np.random.choice(np.arange(self.actions))
                if node.children[action].is_leaf:
                    node, v = self.expand_node(node, action, node.player)
                    node.backup(v)
                    node.v = v
                    break
                else:
                    node = node.children[action]
        self.root_node.remove_noise()  # Don't think this is necessary?

    def load_state_dict(self, state_dict, target=False):
        self.evaluator.load_state_dict(state_dict)

    # determines when a neural net has enough data to train
    @property
    def ready(self):
        # Hard code value for the moment
        return len(self.memory) >= self.min_memory and self.update_nn

    def state_dict(self):
        return self.evaluator.state_dict()

    def update_target_net(self):
        # No target net so pass
        pass

    def deduplicate(self):
        self.memory.deduplicate('state', ['actual_val', 'tree_probs'], Move)

    def train(self, train_state=True):
        # Sets training true/false
        return self.evaluator.train(train_state)

    def evaluate(self, evaluate_state=False):
        # like train - sets evaluate state
        self.evaluating = evaluate_state


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


class ConvNetConnect4(nn.Module):
    def __init__(self, width=7, height=6, action_size=7, default_kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=default_kernel_size, stride=1, padding=1,
                               bias=True)  # Deal with padding?
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=default_kernel_size, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=default_kernel_size, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=default_kernel_size, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=default_kernel_size, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=default_kernel_size, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 64, kernel_size=default_kernel_size, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=default_kernel_size, stride=1, padding=1):
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
        s = s.view(-1, 7, 6)
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
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.bn6(self.conv6(x)))

        # x = x.view(x.size(0), -1)

        policy = F.leaky_relu(self.policy_bn(self.conv_policy(x))).view(x.size(0), -1)
        policy = self.policy_dropout(policy)
        # policy = F.dropout(policy, p=0.3, training=True)  # change training method
        policy = self.softmax(self.linear_policy(policy))

        value = F.leaky_relu(self.value_bn(self.conv_value(x))).view(x.size(0), -1)
        value = self.value_dropout(value)
        # value = F.dropout(value, p=0.3, training=True)  # change training method
        value = F.leaky_relu(self.fc_value(value))
        value = torch.tanh(self.linear_output(value))

        return policy, value
