import datetime
import os
from os import listdir
from os.path import isfile, join

import torch
from torch import multiprocessing
from games.algos.mcts import MCTreeSearch, ConvNetConnect4
# from games.algos.q import EpsilonGreedy, QConvConnect4
from games.connect4.onesteplookahead import OnestepLookahead
from games.algos.self_play_parallel import SelfPlayScheduler
from games.connect4.connect4env import Connect4Env
from games.general.manual import ManualPlay

save_dir = 'saves__c4mtcs_par'


def manual_play():
    env = Connect4Env()

    network = ConvNetConnect4()
    network.share_memory()

    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(temperature_cutoff=3, iterations=400, min_memory=20000,
                         memory_size=50000,
                         env_gen=Connect4Env,
                         evaluator=network)

    opposing_policy = policy_gen(**policy_kwargs)

    folders = [
        join(save_dir, f)
        for f in listdir(os.path.join(save_dir))
        if not isfile(join(save_dir, f))
    ]
    recent_folder = max(folders)

    saves = [
        join(recent_folder, f)
        for f in listdir(os.path.join(recent_folder))
        if isfile(join(recent_folder, f)) and os.path.split(f)[1].startswith("model")
    ]

    recent_file = max(saves)
    opposing_policy.load_state_dict(torch.load(recent_file), target=True)
    opposing_policy.evaluate(True)

    manual = ManualPlay(env, opposing_policy)
    manual.play()


if __name__ == "__main__":
    manual_play()
