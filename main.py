import argparse
import os

from torch import multiprocessing

from games.algos.base_worker import recent_save_file
from games.algos.elo import Elo, ModelDatabase
from games.algos.self_play_parallel import SelfPlayScheduler
from games.connect4 import connect4config
from games.connect4.connect4env import Connect4Env
from games.general.base_env import BaseEnv
from games.general.base_model import ModelContainer
from games.general.hardcoded_players import OneStepLookahead, Random
from games.general.modules import ResidualTower
from games.tictactoe import tictactoeconfig
from games.tictactoe.tictactoe_env import TicTacToeEnv

game_dict = {"connect4": Connect4Env, "tictactoe": TicTacToeEnv}
config_dict = {"connect4": connect4config, "tictactoe": tictactoeconfig}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", help="command", choices=["observe", "calculate_elo", "compare_models", "manual", "train"]
    )
    parser.add_argument("--p", nargs="*", help="players")
    parser.add_argument("--b", nargs="*", help="board size", dest="board_size")
    parser.add_argument(
        "--g", dest="game", help="game - connect4 or tictactoe", default="connect4", choices=["connect4", "tictactoe"]
    )
    parser.add_argument("--o", dest="opponent")
    parser.add_argument("--c", dest="config")
    parser.add_argument("--w", dest="win_config")
    parser.add_argument("--n", dest="name")
    args = parser.parse_args()
    board_args = args.board_size or []
    board_args = [int(x) for x in board_args]
    env: BaseEnv = game_dict[args.game](*board_args)

    md = ModelDatabase(env.variant_string())
    elo = Elo(md)

    if args.command == "observe":
        player_1, player_2 = args.p
        print(player_1, player_2)
        md.observe(player_1, player_2)
    elif args.command == "calculate_elo":
        elo.calculate_elo()
    elif args.command == "compare_models":
        players = args.p
        if not players:
            elo.compare_all()
        else:
            elo.compare_models(*players)
    elif args.command == "manual":
        md.manual_play(args.opponent)
    elif args.command == "train":
        policy = _get_model(args.config, md, env, args.game)
        opposing_policy = _get_model(args.opponent, md, env, args.game)
        train(env, policy, opposing_policy, elo, args.name)


def _get_model(model_name: str, model_database: ModelDatabase, env: BaseEnv, game: str) -> ModelContainer:
    base_model_dict = {"random": Random, "lookahead": OneStepLookahead}
    if model_name in model_database.model_shelf:
        return model_database.get_model(model_name)
    elif model_name in base_model_dict:
        return ModelContainer(base_model_dict[model_name], policy_kwargs=dict(env=env))
    elif getattr(config_dict[game], model_name):
        model: ModelContainer = getattr(config_dict[game], model_name)
        model.set_env(env)
        network = ResidualTower.from_env(env, filter_factor=32, num_blocks=15)
        network.share_memory()
        model.set_network(network)
        return model
    raise ModuleNotFoundError


def train(game: BaseEnv, policy_container: ModelContainer, opponent: ModelContainer, elo: Elo, name):
    save_dir = f"saves__{game.variant_string()}-{name}"
    try:
        os.mkdir(save_dir)
    except Exception:
        pass

    self_play = SelfPlayScheduler(
        env=game,
        policy_container=policy_container,
        evaluation_policy_container=opponent,
        initial_games=20,
        epoch_length=750,
        evaluation_games=75,
        save_dir=save_dir,
        stagger=True,
        lr=0.005,
        deduplicate=False,
        update_delay=0.01,
    )
    self_play.train_model(20, resume_memory=True, resume_model=True, threads_per_worker=1)
    save_file = recent_save_file(save_dir, self_play.start_time, False, "model")
    policy_container.set_network(self_play.network)
    policy_container.load_state_dict(save_file)
    elo.model_database.add_model(name, policy_container)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # May have to modify depending on environment
    main()
