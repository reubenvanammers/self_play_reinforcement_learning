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

try:
    save_dir = "saves__c4mtcs_par"
    os.mkdir(save_dir)
except Exception:
    pass


def run_training():
    multiprocessing.set_start_method('spawn')

    env = Connect4Env()

    network = ConvNetConnect4()
    network.share_memory()

    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(temperature_cutoff=3, iterations=400, min_memory=20000,
                         memory_size=500000,
                         env_gen=Connect4Env,
                         evaluator=network)

    opposing_policy_gen = OnestepLookahead
    opposing_policy_args = []
    opposing_policy_kwargs = dict(env_gen=Connect4Env, player=-1)

    # policy = EpsilonGreedy(QConvTicTacToe(env, buffer_size=5000, batch_size=64), 0.1)

    self_play = SelfPlayScheduler(
        env_gen=Connect4Env,
        policy_gen=policy_gen,
        opposing_policy_gen=opposing_policy_gen,
        policy_args=policy_args,
        policy_kwargs=policy_kwargs,
        opposing_policy_args=opposing_policy_args,
        opposing_policy_kwargs=opposing_policy_kwargs,
        initial_games=5000,
        epoch_length=500,
        save_dir=save_dir,
    )

    # policy = MCTreeSearch(ConvNetTicTacToe(3, 3, 9), TicTacToeEnv, temperature_cutoff=1, iterations=200, min_memory=64)
    # opposing_policy = EpsilonGreedy(
    #     QConvTicTacToe(env), 1
    # )  # Make it not act greedily for the moment- exploration Acts greedily
    # self_play = SelfPlay(policy, opposing_policy, env=env, swap_sides=True)
    self_play.train_model(100, resume=False, num_workers=2)
    print("Training Done")

    saved_name = os.path.join(save_dir, datetime.datetime.now().isoformat())
    torch.save(self_play.policy.q.policy_net.state_dict(), saved_name)


# def resume_self_play():
#     env = TicTacToeEnv()
#     saves = [f for f in listdir(save_dir) if isfile(join(save_dir, f))]
#     recent_file = max(saves)
#     policy = EpsilonGreedy(QConvTicTacToe(env), 0)
#     opposing_policy = EpsilonGreedy(QConvTicTacToe(env), 1)
#     self_play = SelfPlay(policy, opposing_policy)
#     policy.q.policy_net.load_state_dict(torch.load(join(save_dir, recent_file)))
#     self_play.evaluate_policy(100)


def interactive_play():
    pass


if __name__ == "__main__":
    run_training()
    # resume_self_play()

    # self_play.evaluate_policy(1000)
    # # self_play.policy.epsilon = 0
    # self_play.opposing_policy = EpsilonGreedy(QLinear(env), 0.1)
    # self_play.evaluate_policy(1000)
