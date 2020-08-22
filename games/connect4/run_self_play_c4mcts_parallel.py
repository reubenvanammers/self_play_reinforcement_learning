import datetime
import os
from os import listdir
from os.path import isfile, join
from games.algos.base_model import ModelContainer

import torch
from torch import multiprocessing
from games.algos.mcts import MCTreeSearch

from games.connect4.modules import ConvNetConnect4, DeepConvNetConnect4

# from games.algos.q import EpsilonGreedy, QConvConnect4
from games.connect4.hardcoded_players import OnestepLookahead
from games.algos.self_play_parallel import SelfPlayScheduler
from games.connect4.connect4env import Connect4Env

try:
    save_dir = "saves__c4mtcs_par"
    os.mkdir(save_dir)
except Exception:
    pass


def run_training():
    env = Connect4Env()

    network = DeepConvNetConnect4()
    network.share_memory()

    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(iterations=400, min_memory=100000, memory_size=20000, env_gen=Connect4Env,
                         # evaluator=network,
                         )
    policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)

    self_play = True
    if self_play:
        # opposing_network = DeepConvNetConnect4()
        opposing_policy_gen = MCTreeSearch
        opposing_policy_args = []
        opposing_policy_kwargs = dict(
            iterations=400, min_memory=100000, memory_size=20000, env_gen=Connect4Env,
            # evaluator=network
        )
        opposing_policy_container = ModelContainer(policy_gen=opposing_policy_gen, policy_kwargs=opposing_policy_kwargs)

        # evaluation_policy_gen = OnestepLookahead
        # evaluation_policy_args = []
        # evaluation_policy_kwargs = dict(env_gen=Connect4Env, player=-1)
        # evaluation_policy_container = ModelContainer(
        #     policy_gen=evaluation_policy_gen, policy_kwargs=evaluation_policy_kwargs
        # )


        evaluation_network = ConvNetConnect4()

        evaluation_state_dict = torch.load(
            '/Users/reuben/PycharmProjects/reinforcement_learning/saves/model-2020-05-13T22_26_13.689443_4000',map_location=torch.device('cpu'))["model"]

        evaluation_policy_gen = MCTreeSearch
        evaluation_policy_args = []
        evaluation_policy_kwargs = dict(
            iterations=400, min_memory=20000, memory_size=50000, env_gen=Connect4Env, evaluator=evaluation_network,
            starting_state_dict=evaluation_state_dict
        )
        evaluation_policy_container = ModelContainer(
            policy_gen=evaluation_policy_gen, policy_kwargs=evaluation_policy_kwargs
        )
        check = evaluation_policy_container.setup()
        print(check)

    else:
        pass
        # opposing_policy_gen = OnestepLookahead
        # opposing_policy_args = []
        # opposing_policy_kwargs = dict(env_gen=Connect4Env, player=-1)
        # evaluation_policy_gen = None
        # evaluation_policy_args = []
        # evaluation_policy_kwargs = {}

    # policy = EpsilonGreedy(QConvTicTacToe(env, buffer_size=5000, batch_size=64), 0.1)

    self_play = SelfPlayScheduler(
        env_gen=Connect4Env,
        network=network,
        policy_container=policy_container,
        opposing_policy_container=opposing_policy_container,
        evaluation_policy_container=evaluation_policy_container,
        initial_games=0,
        epoch_length=500,
        save_dir=save_dir,
        self_play=self_play,
        stagger=False,
        lr=0.0003
    )

    # policy = MCTreeSearch(ConvNetTicTacToe(3, 3, 9), TicTacToeEnv, temperature_cutoff=1, iterations=200, min_memory=64)
    # opposing_policy = EpsilonGreedy(
    #     QConvTicTacToe(env), 1
    # )  # Make it not act greedily for the moment- exploration Acts greedily
    # self_play = SelfPlay(policy, opposing_policy, env=env, swap_sides=True)
    self_play.train_model(100, resume_memory=True, resume_model=True)
    print("Training Done")

    saved_name = os.path.join(save_dir, datetime.datetime.now().isoformat())
    torch.save(self_play.policy.q.policy_net.state_dict(), saved_name)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_training()
