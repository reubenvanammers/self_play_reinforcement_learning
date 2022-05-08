from multiprocessing import Queue

import torch
from torch import nn

from games.general.base_env import BaseEnv
from rl_utils.memory import Memory


class BasePlayer:
    def __call__(self, s):
        raise NotImplementedError

    def reset(self, player=1):
        raise NotImplementedError

    def play_action(self, action, player):
        raise NotImplementedError

    # TODO - Maybe refactor?
    def train(self, train_state):
        pass

    def evaluate(self, evaluate_state=False):
        pass


# Must have a network to be updated
class TrainableModel:
    def __init__(self, memory_queue: Queue, memory_size: int, *args, **kwargs):
        self.memory = self.create_memory(memory_size)
        self.create_memory(memory_size)
        self.memory_queue = memory_queue

    def create_memory(self, memory_size):
        return Memory(memory_size)

    def load_state_dict(self, state_dict, target=False):
        raise NotImplementedError

    def update(self, s, a, r, done, next_s):
        self.push_to_queue(s, a, r, done, next_s)
        self.pull_from_queue()
        if self.ready:
            self.update_from_memory()

    def update_from_memory(self):
        raise NotImplementedError

    @property
    def ready(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def train(self, train_state):
        raise NotImplementedError

    def evaluate(self, evaluate_state=False):
        pass

    def pull_from_queue(self):
        while not self.memory_queue.empty():
            experience = self.memory_queue.get()
            self.memory.add(experience)

    def push_to_queue(self, s, a, r, done, next_s):
        raise NotImplementedError

    def deduplicate(self):
        pass


class Policy(TrainableModel, BasePlayer):
    pass


class ModelContainer:
    def __init__(self, policy_gen, policy_args=[], policy_kwargs={}):
        self.policy_gen = policy_gen
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs

    def setup(self, **kwargs) -> BasePlayer:
        if "evaluator" in self.policy_kwargs:
            self.policy_kwargs["network"] = self.policy_kwargs.pop("evaluator")
        return self.policy_gen(*self.policy_args, **self.policy_kwargs, **kwargs)

    def load_state_dict(self, save_file):
        checkpoint = torch.load(save_file)
        self.policy_kwargs["network"].load_state_dict(checkpoint["model"])

    def set_env(self, env: BaseEnv):
        self.policy_kwargs["env"] = env
        return self

    def set_network(self, network: nn.Module):
        self.policy_kwargs["network"] = network
        return self
