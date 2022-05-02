import torch
from elo import Elo, ModelDatabase
from torch import multiprocessing

from games.algos.base_model import ModelContainer
from games.algos.mcts import MCTreeSearch
from games.connect4.connect4env import Connect4Env
from games.connect4.modules import (ConvNetConnect4, DeepConvNetConnect4,
                                    ResidualTower)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # May have to modify depending on environment

    # policy_gen = MCTreeSearch
    # policy_args = []
    # network = ResidualTower(width=7, height=6,action_size=7,num_blocks=20)
    # checkpoint = torch.load("/home/reuben/PycharmProjects/self_play_reinforcement_learning/games/connect4/saves__c4mtcs_par/2021-06-22T22:17:48.101873/model-2021-06-23T09:05:06.751248:64000")
    # network.load_state_dict(checkpoint["model"])
    #
    # policy_kwargs = dict(iterations=800, min_memory=25000, memory_size=300000, env_gen=Connect4Env, batch_size=128,network=network)

    md = ModelDatabase()
    elo = Elo(md)
    md.observe('mcts1', 'test-big-res')
    # container = ModelContainer(MCTreeSearch, policy_kwargs=policy_kwargs)
    # md.add_model('test-big-res', container)

    # elo.compare_models('test-big-res', '15layer-num1', 'cloudmcts')
    # elo.calculate_elo()
