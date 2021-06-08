import concurrent.futures
import copy
import logging
import random
import threading
import time
import traceback
from collections import namedtuple
from multiprocessing import Queue

import numpy as np
import torch
from anytree import NodeMixin
from torch import nn
from torch.functional import F

from games.algos.base_model import BaseModel
from games.algos.inference_proxy import InferenceProxy
from rl_utils.flat import MSELossFlat
from rl_utils.memory import Memory
from rl_utils.weights import init_weights

Move = namedtuple("Move", ("state", "actual_val", "tree_probs"),)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCNode(NodeMixin):
    # Represents an action of a Monte Carlo Search Tree

    def __init__(
        self, state=None, n=0, w=0, p=0, x=0.25, parent=None, cpuct=4, player=1, v=None, valid=True,
    ):
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
        self._valid = valid  # Whether an action is valid - don't play moves that you cannot
        self.cpuct = cpuct  # exploration factor

        # Flag for threading
        self.in_use = False
        self.virtual_loss = 0

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
    def q(self,):  # Attractiveness of a node from player ones pespective - average of downstream results
        n_eff = self.n + self.virtual_loss
        return (self.w - self.virtual_loss) / n_eff if n_eff else 0

    @property  # effective p - p if noise isn't active, otherwise adds noise factor
    def p_eff(self):
        if self.noise_active:
            return self.p_noise * self.x + self.p * (1 - self.x)
        else:
            return self.p

    @property
    def u(self,):  # Factor to encourage exploration - higher values of cpuct increase exploration
        return self.cpuct * self.p_eff * np.sqrt(self.parent.n) / (1 + self.n)

    @property
    def select_prob(self,):  # TODO check if this works? might need to be ther other way round
        return (
            -1 * self.player * self.q + self.u
        )  # -1 is due to that this calculated from the perspective of the parent node, which has an opposite player

    @property
    def valid(self):
        return self._valid and not self.in_use

    def remove_virtual_loss(self):
        self.virtual_loss -=1
        assert self.virtual_loss >= 0

    def backup(self, v):
        self.w += v
        self.n += 1
        # self.virtual_loss -= 1
        # try:
        #     assert self.virtual_loss >= 0
        # except AssertionError:
        #     print(f"{self.virtual_loss}")
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


class MCTreeSearch(BaseModel):
    root_node: MCNode

    def __init__(
        self,
        network,
        env_gen,
        optim=None,
        memory_queue=None,
        iterations=100,
        temperature_cutoff=5,
        batch_size=64,
        memory_size=200000,
        min_memory=20000,
        update_nn=True,
        starting_state_dict=None,
        # threading=True,
    ):
        self.iterations = iterations
        self.network = network.to(device)
        self.env_gen = env_gen
        self.optim = optim
        self.env = env_gen()
        self.root_node = None
        self.reset()
        self.update_nn = update_nn
        self.starting_state_dict = starting_state_dict

        self.memory_queue = memory_queue
        self.temp_memory = []
        self.memory = Memory(memory_size)
        self.min_memory = min_memory
        self.temperature_cutoff = temperature_cutoff
        self.actions = self.env.action_space.n

        self.evaluating = False
        self.threading = isinstance(self.network, InferenceProxy)

        self.batch_size = batch_size

        if self.starting_state_dict:
            # print("laoding [sic] state dict in mcts")
            self.load_state_dict(self.starting_state_dict)

    def reset(self, player=1):
        base_state = self.env.reset()
        probs, v = self.network(base_state)
        self._set_root(MCNode(state=base_state, v=v, player=player))
        self.root_node.create_children(probs, self.env.valid_moves())
        self.moves_played = 0
        self.temp_memory = []

        return base_state

    ## Ignores the inputted state for the moment. Produces the correct action, and changes the root node appropriately
    # TODO might want to do a check on the state to make sure it is consistent
    def __call__(self, s):  # not using player
        move = self._search_and_play()
        return move

    def _search_and_play(self):
        final_temp = 1
        temperature = 1 if self.moves_played < self.temperature_cutoff else final_temp
        self.search()
        move = self._play(temperature)
        return move

    def play_action(self, action, player):
        self._set_node(action)

    def _set_root(self, node):
        if self.root_node:
            self.root_node.active_root = False
        self.root_node = node
        self.root_node.active_root = True

    def _prune(self, action):
        self._set_root(self.root_node.children[action])
        self.root_node.parent.children = [self.root_node]

    def _set_node(self, action):
        node = self.root_node.children[action]
        if self.root_node.children[action].n == 0:
            # TODO check if leaf and n > 0??
            # TODO might not backup??????????
            node, v = self._expand_node(self.root_node, action, self.root_node.player)
            node.backup(v)
            node.v = v
        self._set_root(node)

    def update(self, s, a, r, done, next_s):
        self.push_to_queue(s, a, r, done, next_s)
        self.pull_from_queue()
        if self.ready:
            self.update_from_memory()

    def pull_from_queue(self):
        while not self.memory_queue.empty():
            # logging.info(f"queue size in policy is {self.memory_queue.qsize()}")
            experience = self.memory_queue.get()
            experience = Move(*[experience[i].to(device,) for i in range(3)])
            self.memory.add(experience)
            # logging.info(f"memory size is  is {len(self.memory)}")

    def push_to_queue(self, s=None, a=None, r=None, done=None, next_s=None):
        # Push memory of the game to the memory queue with the actual result of the game
        if done:
            # logging.info('pushing to queue')
            for experience in self.temp_memory:
                # logging.info(self.memory_queue.qsize())
                # logging.info('pushing experience to queue')
                # experience = experience._replace(actual_val=torch.tensor(r).float().to(device))
                experience = experience._replace(actual_val=torch.tensor(r).float())
                self.memory_queue.put(experience)
            self.temp_memory = []

    def loss(self, batch):
        batch_t = Move(*zip(*batch))  # transposed batch
        s, actual_val, tree_probs = batch_t
        s_batch = torch.stack(s)
        net_probs_batch, predict_val_batch = self.network.forward(s_batch)
        predict_val_batch = predict_val_batch.view(-1)
        actual_val_batch = torch.stack(actual_val)
        tree_probs_batch = torch.stack(tree_probs)

        c = MSELossFlat(floatify=True)
        value_loss = c(predict_val_batch, actual_val_batch)

        prob_loss = -(net_probs_batch.log() * tree_probs_batch).sum() / net_probs_batch.size()[0]

        loss = value_loss + prob_loss
        return loss

    def update_from_memory(self):
        if len(self.memory) < self.batch_size:
            logging.info(
                f"skipping due to not enough memory have {len(self.memory)} objects with batch size {self.batch_size}"
            )
            print("skipping due to not enogugh memory")
            time.sleep(1)
            return
        batch = self.memory.sample(self.batch_size)

        loss = self.loss(batch)

        self.optim.zero_grad()

        # if APEX_AVAILABLE: keep just in case want to use amp
        if False:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # for param in self.evaluator.parameters():  # see if this ends up doing anything - should just be relu
        #     param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def _play(self, temp=0.05):
        if self.evaluating:  # Might want to just make this greedy
            temp = temp / 20  # More likely to choose higher visited nodes

        play_probs = [child.play_prob(temp) for child in self.root_node.children]
        play_probs = play_probs / sum(play_probs)

        try:
            action = np.random.choice(self.actions, p=play_probs)
        except Exception:
            logging.info(f"action exception, actions{self.actions} p = {play_probs}")
            logging.info(traceback.format_exc())

        self.moves_played += 1

        self.temp_memory.append(Move(torch.tensor(self.root_node.state), None, torch.tensor(play_probs).float(),))
        return action

    def _expand_node(self, parent_node, action, player=1):
        env = self.env_gen()
        env.set_state(copy.copy(parent_node.state))
        s, r, done, _ = env.step(action, player=player)
        r = r * player
        if done:
            v = r
            child_node = parent_node.children[action]
        else:
            probs, v = self.network(s, parent_node.player)
            child_node = parent_node.children[action]
            child_node.create_children(probs, env.valid_moves())
            assert child_node.children
        child_node.state = s
        return child_node, v

    def search(self):
        if self.evaluating:
            self.root_node.add_noise()  # Might want to remove this in evaluation?
        else:
            self.root_node.add_noise()
        if self.threading:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                [executor.submit(self.search_node) for _ in range(self.iterations)]
                executor.shutdown()
        else:
            for i in range(self.iterations):
                self.search_node()
        if self.evaluating:
            self.root_node.remove_noise()  # Don't think this is necessary?
        else:
            self.root_node.remove_noise()

    def search_node(self):
        node = self.root_node
        node_list = []
        while True:
            node_list.append(node)
            node.virtual_loss += 1
            select_probs = [
                child.select_prob if child.valid else -10000000000 for child in node.children
            ]  # real big negative number
            if all(i < -100000 for i in select_probs):
                # TODO check if this causes any problems - most of the time if this is happening its getting close to the
                # end of the game. Caused because all threads are currently active
                # May want to move up valid checking one level? or just a wait
                # logging.info("all states currently in use")
                return
            action = np.argmax(select_probs + 0.000001 * np.random.rand(self.actions))
            child_node = node.children[action]
            if child_node.is_leaf:
                child_node.in_use = True
                node, v = self._expand_node(node, action, node.player)
                node.backup(v)
                node.v = v
                node.in_use = False
                [n.remove_virtual_loss() for n in node_list]
                break
            else:
                node = child_node

    def load_state_dict(self, state_dict, target=False):
        self.network.load_state_dict(state_dict)

    # determines when a neural net has enough data to train
    @property
    def ready(self):
        # Hard code value for the moment
        return len(self.memory) >= self.min_memory and self.update_nnevaluator

    def state_dict(self):
        return self.network.state_dict()

    def update_target_net(self):
        # No target net so pass
        pass

    def deduplicate(self):
        self.memory.deduplicate("state", ["actual_val", "tree_probs"], Move)

    def train(self, train_state=True):
        # Sets training true/false
        return self.network.train(train_state)

    def evaluate(self, evaluate_state=False):
        # like train - sets evaluate state
        self.evaluating = evaluate_state
