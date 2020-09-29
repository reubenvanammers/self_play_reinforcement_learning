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


def set_parameters():
    opposing_state_dict = torch.load(
        "/Users/reuben/PycharmProjects/reinforcement_learning/games/connect4/saves__c4mtcs_par/2020-05-12T17:28:34.659107/model-2020-05-12T18:39:28.650785:210"
    )["model"]

    parameters = [
        {
            "policy_kwargs": dict(iterations=400, memory_size=50000, env_gen=Connect4Env),
            "evaluator": ConvNetConnect4,
            "evaluation_policy_kwargs": dict(
                iterations=400, memory_size=50000, env_gen=Connect4Env, starting_state_dict=opposing_state_dict,
            ),
            "evaluation_evaluator": ConvNetConnect4,
        },
        {
            "lr": 0.01,
            "policy_kwargs": dict(iterations=400, memory_size=50000, env_gen=Connect4Env),
            "evaluator": ConvNetConnect4,
            "evaluation_policy_kwargs": dict(
                iterations=400, memory_size=50000, env_gen=Connect4Env, starting_state_dict=opposing_state_dict,
            ),
            "evaluation_evaluator": ConvNetConnect4,
        },
        {
            "policy_kwargs": dict(iterations=100, memory_size=50000, env_gen=Connect4Env),
            "evaluator": ConvNetConnect4,
            "evaluation_policy_kwargs": dict(
                iterations=400, memory_size=50000, env_gen=Connect4Env, starting_state_dict=opposing_state_dict,
            ),
            "evaluation_evaluator": ConvNetConnect4,
        },
        {
            "policy_kwargs": dict(iterations=100, memory_size=20000, env_gen=Connect4Env),
            "evaluator": ConvNetConnect4,
            "evaluation_policy_kwargs": dict(
                iterations=400, memory_size=50000, env_gen=Connect4Env, starting_state_dict=opposing_state_dict,
            ),
            "evaluation_evaluator": ConvNetConnect4,
            "stagger": True,
        },
    ]
    for parameter in parameters:
        run_training(**parameter)


def run_training(
    policy_kwargs,
    self_play=True,
    evaluator=None,
    initial_games=100,
    epoch_length=500,
    num_epochs=1,
    stagger=False,
    evaluation_policy_kwargs=None,
    evaluation_evaluator=None,
    lr=0.001,
):
    policy_gen = MCTreeSearch
    policy_args = []
    network = evaluator()

    if self_play:

        opposing_policy_gen = MCTreeSearch
        opposing_policy_args = []
        opposing_policy_kwargs = policy_kwargs

        evaluation_policy_gen = MCTreeSearch
        evaluation_policy_args = []
        evaluation_policy_kwargs = evaluation_policy_kwargs
        evaluation_policy_kwargs["evaluator"] = evaluation_evaluator()

    else:
        opposing_policy_gen = OnestepLookahead
        opposing_policy_args = []
        opposing_policy_kwargs = dict(env_gen=Connect4Env, player=-1)
        evaluation_policy_gen = None
        evaluation_policy_args = []
        evaluation_policy_kwargs = {}

    self_play = SelfPlayScheduler(
        network=network,
        env_gen=Connect4Env,
        policy_gen=policy_gen,
        opposing_policy_gen=opposing_policy_gen,
        policy_args=policy_args,
        policy_kwargs=policy_kwargs,
        opposing_policy_args=opposing_policy_args,
        opposing_policy_kwargs=opposing_policy_kwargs,
        initial_games=initial_games,
        epoch_length=epoch_length,
        save_dir=save_dir,
        self_play=self_play,
        lr=lr,
        evaluation_policy_gen=evaluation_policy_gen,
        evaluation_policy_args=evaluation_policy_args,
        evaluation_policy_kwargs=evaluation_policy_kwargs,
        stagger=stagger,
    )

    self_play.train_model(num_epochs, resume_memory=False, resume_model=False)
    print("Training Done")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    set_parameters()
