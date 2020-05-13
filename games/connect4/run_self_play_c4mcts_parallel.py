import datetime
import os
from os import listdir
from os.path import isfile, join

import torch
from torch import multiprocessing
from games.algos.mcts import MCTreeSearch

from games.connect4.modules import ConvNetConnect4, DeepConvNetConnect4
# from games.algos.q import EpsilonGreedy, QConvConnect4
from games.connect4.onesteplookahead import OnestepLookahead
from games.algos.self_play_parallel import SelfPlayScheduler
from games.connect4.connect4env import Connect4Env

try:
    save_dir = "saves__c4mtcs_par"
    os.mkdir(save_dir)
except Exception:
    pass


def run_training():
    # multiprocessing.set_start_method('spawn')

    env = Connect4Env()

    network = ConvNetConnect4()
    network.share_memory()

    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(iterations=200, min_memory=20000, memory_size=50000, env_gen=Connect4Env, evaluator=network, )

    self_play = True
    if self_play:
        # opposing_network = DeepConvNetConnect4()
        opposing_policy_gen = MCTreeSearch
        opposing_policy_args = []
        opposing_policy_kwargs = dict(
            iterations=200, min_memory=20000, memory_size=50000, env_gen=Connect4Env, evaluator=network,
        )


        evaluation_policy_gen = OnestepLookahead
        evaluation_policy_args = []
        evaluation_policy_kwargs = dict(env_gen=Connect4Env, player=-1)
        # opposing_state_dict = torch.load(
        #     '/Users/reuben/PycharmProjects/reinforcement_learning/games/connect4/saves__c4mtcs_par/2020-05-12T17:28:34.659107/model-2020-05-12T18:39:28.650785:210')
        #
        # evaluation_policy_gen = MCTreeSearch
        # evaluation_policy_args = []
        # evaluation_policy_kwargs = dict(
        #     iterations=200, min_memory=20000, memory_size=50000, env_gen=Connect4Env, evaluator=network,
        #     starting_state_dict=opposing_state_dict
        # )


    else:
        opposing_policy_gen = OnestepLookahead
        opposing_policy_args = []
        opposing_policy_kwargs = dict(env_gen=Connect4Env, player=-1)
        evaluation_policy_gen = None
        evaluation_policy_args = []
        evaluation_policy_kwargs = {}

    # policy = EpsilonGreedy(QConvTicTacToe(env, buffer_size=5000, batch_size=64), 0.1)

    self_play = SelfPlayScheduler(
        env_gen=Connect4Env,
        policy_gen=policy_gen,
        opposing_policy_gen=opposing_policy_gen,
        policy_args=policy_args,
        policy_kwargs=policy_kwargs,
        opposing_policy_args=opposing_policy_args,
        opposing_policy_kwargs=opposing_policy_kwargs,
        initial_games=50,
        epoch_length=200,
        save_dir=save_dir,
        self_play=self_play,
        evaluation_policy_gen=evaluation_policy_gen,
        evaluation_policy_args=evaluation_policy_args,
        evaluation_policy_kwargs=evaluation_policy_kwargs,
    )

    # policy = MCTreeSearch(ConvNetTicTacToe(3, 3, 9), TicTacToeEnv, temperature_cutoff=1, iterations=200, min_memory=64)
    # opposing_policy = EpsilonGreedy(
    #     QConvTicTacToe(env), 1
    # )  # Make it not act greedily for the moment- exploration Acts greedily
    # self_play = SelfPlay(policy, opposing_policy, env=env, swap_sides=True)
    self_play.train_model(100, resume_memory=False, resume_model=False)
    print("Training Done")

    saved_name = os.path.join(save_dir, datetime.datetime.now().isoformat())
    torch.save(self_play.policy.q.policy_net.state_dict(), saved_name)


if __name__ == "__main__":
    run_training()
