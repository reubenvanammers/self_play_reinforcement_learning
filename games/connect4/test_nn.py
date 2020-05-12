import datetime
import os
from os import listdir
from os.path import isfile, join

import random

import torch
from torch import multiprocessing
from games.algos.mcts import MCTreeSearch, ConvNetConnect4

# from games.algos.q import EpsilonGreedy, QConvConnect4
from games.connect4.onesteplookahead import OnestepLookahead
from games.algos.self_play_parallel import SelfPlayScheduler
from games.connect4.connect4env import Connect4Env
import pickle
from collections import deque
from torch.utils.data import random_split, Dataset, DataLoader
from rl_utils.weights import init_weights

import datetime

start_time = datetime.datetime.now().isoformat()

save_dir = "test_nn"
os.mkdir(save_dir)
os.mkdir(os.path.join(save_dir, start_time))


class ListDataset(Dataset):
    def __init__(self, list):
        self.list = list

    def __len__(self):
        return len(self.list)


EPOCH_LENGTH = 5000
NUM_EPOCHS = 1000


def run():
    MEM_FILE = "/Users/reuben/PycharmProjects/reinforcement_learning/games/connect4/saves__c4mtcs_par/2020-04-26T22:06:52.551894/memory-2020-04-27T08:40:07.492133:134014"

    with open(MEM_FILE, "rb") as f:
        memory = pickle.load(f)

    VAL_AMOUNT = len(memory) // 10
    TRAIN_AMOUNT = len(memory) - VAL_AMOUNT

    dataset = list(memory.buffer)
    train_indexes = random.sample(range(len(dataset)), TRAIN_AMOUNT)
    val_indexes = list(set(range(len(dataset))) - set(train_indexes))

    train_data = [dataset[i] for i in train_indexes]
    val_data = [dataset[i] for i in val_indexes]

    memory.buffer = deque(train_data)

    # train_set, val_set = random_split(dataset, [TRAIN_AMOUNT, VAL_AMOUNT])

    network = ConvNetConnect4()
    network.apply(init_weights)
    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(
        temperature_cutoff=3,
        iterations=400,
        min_memory=20000,
        memory_size=500000,
        env_gen=Connect4Env,
        evaluator=network,
        memory_queue=None,
    )
    policy = policy_gen(*policy_args, **policy_kwargs)
    policy.memory = memory

    # dataloader = DataLoader(train_set, batch_size=64)

    loss = policy.loss(val_data)
    print(loss)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        policy.optim, "min", patience=30, factor=0.2, verbose=True, min_lr=0.00001
    )

    for epoch in range(NUM_EPOCHS):
        saved_name = os.path.join(
            save_dir, start_time, "model-" + datetime.datetime.now().isoformat() + ":" + str(EPOCH_LENGTH * epoch),
        )
        torch.save(policy.state_dict(), saved_name)  # also save memory

        for i in range(EPOCH_LENGTH):
            policy.update_from_memory()
        print(epoch)
        loss = policy.loss(val_data)
        scheduler.step(loss)
        print(loss)


if __name__ == "__main__":
    run()
