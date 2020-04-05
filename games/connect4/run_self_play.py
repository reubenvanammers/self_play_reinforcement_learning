import datetime
import os
from os import listdir
from os.path import isfile, join

import torch

from games.algos.q import EpsilonGreedy, QConvConnect4, QLinear
from games.algos.self_play import SelfPlay
from games.connect4.connect4env import Connect4Env

save_dir = "saves"


def run_training():
    env = Connect4Env()
    policy = EpsilonGreedy(QConvConnect4(env, buffer_size=10000), 0.1)
    opposing_policy = EpsilonGreedy(
        QConvConnect4(env), 1
    )  # Make it not act greedily for the moment- exploration Acts greedily
    self_play = SelfPlay(policy, opposing_policy, env)
    self_play.train_model(5000, resume=False)
    print("Training Done")

    saved_name = os.path.join(save_dir, datetime.datetime.now().isoformat())
    torch.save(self_play.policy.q.policy_net.state_dict(), saved_name)


def resume_self_play():
    env = Connect4Env()
    saves = [f for f in listdir(save_dir) if isfile(join(save_dir, f))]
    recent_file = max(saves)
    policy = EpsilonGreedy(QLinear(env), 0)
    opposing_policy = EpsilonGreedy(QLinear(env), 0)  # Acts greedily
    self_play = SelfPlay(policy, opposing_policy)
    policy.q.policy_net.load_state_dict(torch.load(join(save_dir, recent_file)))
    self_play.evaluate_policy(100)


def interactive_play():
    pass


if __name__ == "__main__":
    run_training()
    resume_self_play()

    # self_play.evaluate_policy(1000)
    # # self_play.policy.epsilon = 0
    # self_play.opposing_policy = EpsilonGreedy(QLinear(env), 0.1)
    # self_play.evaluate_policy(1000)
