import os

from torch import multiprocessing

from games.algos.base_model import ModelContainer
from games.algos.mcts import MCTreeSearch
from games.algos.self_play_parallel import SelfPlayScheduler
from games.connect4.connect4env import Connect4Env
from games.connect4.hardcoded_players import OnestepLookahead
from games.connect4.modules import ConvNetConnect4, DeepConvNetConnect4

try:
    save_dir = "saves__c4mtcs_par"
    os.mkdir(save_dir)
except Exception:
    pass


def run_training():
    # Can update config for training from here
    env = Connect4Env()

    network = DeepConvNetConnect4()
    network.share_memory()

    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(
        iterations=400,
        min_memory=100000,
        memory_size=20000,
        env_gen=Connect4Env,
        # evaluator=network,
        batch_size=64,
    )
    policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)

    opposing_policy_gen = MCTreeSearch
    opposing_policy_args = []
    opposing_policy_kwargs = dict(
        iterations=400, min_memory=100000, memory_size=20000, env_gen=Connect4Env, batch_size=64
    )
    opposing_policy_container = ModelContainer(policy_gen=opposing_policy_gen, policy_kwargs=opposing_policy_kwargs)

    # Can update evaluation network if wanting to have a more powerful evaluation function, eg after the
    # model has gotten strong enough.
    evaluation_policy_gen = OnestepLookahead
    evaluation_policy_args = []
    evaluation_policy_kwargs = dict(env_gen=Connect4Env, player=-1)
    evaluation_policy_container = ModelContainer(
        evaluation_policy_gen, evaluation_policy_args, evaluation_policy_kwargs
    )

    self_play = SelfPlayScheduler(
        env_gen=Connect4Env,
        network=network,
        policy_container=policy_container,
        opposing_policy_container=opposing_policy_container,
        evaluation_policy_container=evaluation_policy_container,
        initial_games=40,
        epoch_length=500,
        evaluation_games=100,
        save_dir=save_dir,
        self_play=True,
        stagger=False,
        lr=0.0003,
    )

    self_play.train_model(100, resume_memory=False, resume_model=False)
    print("Training Done")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # May have to modify depending on environment
    run_training()
