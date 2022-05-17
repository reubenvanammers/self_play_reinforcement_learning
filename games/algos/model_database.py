import atexit
import os
import shelve

from games.connect4.connect4env import Connect4Env
from games.general.external_play import ManualPlay, View
from games.tictactoe.tictactoe_env import TicTacToeEnv
from rl_utils.memory import Memory


class ModelDatabase:
    """
    This class is used to store the results of different models against each other. Models can be registered and
    persisted, and then given the command to play against eachother. The Elo results can then be calculated,
    assuming that an anchor model (assumed to be random, and set to 0 elo) exists, using the Elo class.

    from games.connect4.hardcoded_players import Random
    model_database = ModelDatabase()
    policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)
    # This persists the model and its arguments to disk
    model_database.add_model("model_1", policy_container)

    random_policy = ModelContainer(policy_gen=Random, policy_kwargs={})
    model_database.add_model("random", random_policy)

    elo = Elo(model_database)
    elo.compare_model("random","model_1") # Plays games between these models
    elo.calculate_elo()

    """

    MODEL_SAVE_FILE = ".ELO_MODEL"
    RESULT_SAVE_FILE = ".ELO_RESULT"
    ELO_VALUE_SAVE_FILE = ".ELO_VALUE"

    GAME_DICT = {
        "connect4": Connect4Env,
        "tictactoe": TicTacToeEnv,
    }

    def __init__(self, game="connect4"):
        path = os.path.dirname(__file__)
        self.model_shelf = shelve.open(os.path.join(path, f"{game}{self.MODEL_SAVE_FILE}"))
        self.result_shelf = shelve.open(os.path.join(path, f"{game}{self.RESULT_SAVE_FILE}"))
        self.elo_value_shelf = shelve.open(os.path.join(path, f"{game}{self.ELO_VALUE_SAVE_FILE}"))

        self.game = game

        self.memory = Memory()

        atexit.register(self._close)

    def _close(self):
        self.model_shelf.close()
        self.result_shelf.close()
        self.elo_value_shelf.close()

    def elos(self):
        return {key: self.elo_value_shelf[key] for key in self.elo_value_shelf.keys()}

    def get_model(self, model_name):
        return self.model_shelf[model_name]

    def add_model(self, name, model_container):
        try:
            if self.model_shelf[name]:
                raise ValueError("Model name already in use")
        except KeyError:
            self.model_shelf[name] = model_container
            print(f"added model {name}")

    def manual_play(self, model_name):
        model = self.model_shelf[model_name].setup()
        model.train(False)
        model.evaluate(True)

        manual = ManualPlay(model.env(), model)
        manual.play()

    def observe(self, model_name, opponent_model_name):

        model = self.model_shelf[model_name].setup()
        opponent_model = self.model_shelf[opponent_model_name].setup()

        model.train(False)
        opponent_model.train(False)
        model.evaluate(True)
        opponent_model.evaluate(True)

        view = View(Connect4Env(), model, opponent_model)
        view.play()


if __name__ == "__main__":
    model_database = ModelDatabase("connect4")
    import sys

    from games.general import base_model, hardcoded_players
    from games.general.modules import BasicBlock, ResidualTower

    # sys.modules['games.algos.base_model'] = base_model
    # sys.modules['games.connect4.modules'].ResidualTower = ResidualTower
    # sys.modules['games.connect4.modules'].BasicBlock = BasicBlock
    # sys.modules['games.connect4.hardcoded_players'] = hardcoded_players
    # hardcoded_players.OnestepLookahead = hardcoded_players.OneStepLookahead
    for model_name in model_database.model_shelf:
        model = model_database.model_shelf[model_name]
        model_database.model_shelf[model_name] = model
    print(model_database.elos())
