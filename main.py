import argparse

from torch import multiprocessing

from games.algos.elo import Elo, ModelDatabase


def main():
    md = ModelDatabase()
    elo = Elo(md)

    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="command")
    parser.add_argument("--p", nargs="*", help="players")

    args = parser.parse_args()
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


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # May have to modify depending on environment
    main()
