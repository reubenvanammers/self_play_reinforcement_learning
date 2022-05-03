import os

from torch import multiprocessing

from games.algos.mcts import MCTreeSearch
from games.algos.model_database import ModelDatabase
from games.algos.self_play_parallel import SelfPlayScheduler
from games.connect4.connect4env import Connect4Env
from games.general.base_model import ModelContainer
from games.general.hardcoded_players import OneStepLookahead
from games.general.modules import ResidualTower

try:
    save_dir = "saves__c4mtcs_par"
    os.mkdir(save_dir)
except Exception:
    pass


def run_training():
    # Can update config for training from here

    # network = DeepConvNetConnect4()
    network = ResidualTower(width=7, height=6, action_size=7, num_blocks=20)
    network.share_memory()

    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(
        iterations=800,
        min_memory=25000,
        memory_size=300000,
        env_gen=Connect4Env,
        batch_size=128,
    )
    policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)

    model_db = ModelDatabase()

    # Can update evaluation network if wanting to have a more powerful evaluation function, eg after the
    # model has gotten strong enough.
    evaluation_policy_gen = OneStepLookahead
    evaluation_policy_args = []
    evaluation_policy_kwargs = dict(env_gen=Connect4Env, player=-1)
    evaluation_policy_container = ModelContainer(
        evaluation_policy_gen, evaluation_policy_args, evaluation_policy_kwargs
    )

    # evaluation_policy_container = model_db.get_model("model_name")

    if evaluation_policy_container.policy_kwargs.get("evaluator"):
        evaluation_network = evaluation_policy_container.policy_kwargs["evaluator"]
        evaluation_network.load_state_dict(evaluation_policy_container.policy_kwargs["starting_state_dict"])
        del evaluation_policy_container.policy_kwargs["evaluator"]
    else:
        evaluation_network = None

    self_play = SelfPlayScheduler(
        env_gen=Connect4Env,
        network=network,
        policy_container=policy_container,
        evaluation_policy_container=evaluation_policy_container,
        initial_games=40,
        epoch_length=1500,
        evaluation_games=150,
        save_dir=save_dir,
        self_play=True,
        stagger=True,
        stagger_mem_step=15000,
        lr=0.005,
        evaluation_network=evaluation_network,
        deduplicate=False,
        update_delay=0.01,
    )
    self_play.train_model(500, resume_memory=True, resume_model=True)
    print("Training Done")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # May have to modify depending on environment
    run_training()
