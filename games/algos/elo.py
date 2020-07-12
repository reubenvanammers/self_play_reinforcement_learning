from games.algos import self_play_parallel
from games.connect4.connect4env import Connect4Env
import shelve
import os
import itertools
import atexit
from games.algos.base_model import ModelContainer


class Elo():
    MODEL_SAVE_FILE = ".ELO_MODEL"
    RESULT_SAVE_FILE = ".ELO_RESULT"

    def __init___(self, env=Connect4Env):
        path = os.path.dirname(__file__)
        self.model_shelf = shelve.open(os.path.join(path, self.MODEL_SAVE_FILE))
        self.result_shelf = shelve.open(os.path.join(path, self.RESULT_SAVE_FILE))

        self.env = env

    @atexit.register
    def _close(self):
        self.model_shelf.close()
        self.result_shelf.close()

    def add_model(self, name, model_container):
        if self.model_shelf[name]:
            raise ValueError("Model name already in use")
        self.model_shelf[name] = model_container

    def compare_models(self, *args):
        combinations = itertools.combinations(args, 2)

        for model_1, model_2 in combinations:
            self._compare(model_1, model_2)

    def _compare(self, model_1, model_2, num_games=100):
        assert model_1 != model_2
        assert "_" not in model_1
        assert "_" not in model_2
        if model_1 > model_2:
            key = f"{model_1}__{model_2}"
        else:
            key = f"{model_2}__{model_1}"
        if key in self.result_shelf:
            old_results = self.result_shelf[key]
        else:
            old_results = {"wins": 0, "draws": 0, "losses": 0}
        new_results = self._get_results(model_1, model_2)
        total_results = {status: new_results[status] + old_results[status] for status in ("wins", "draws", "losses")}
        self.result_shelf[key] = total_results

    def _get_results(self, model_1, model_2, num_games=100):
        # model_1_dict = self.model_shelf[model_1]
        # model_2_dict = self.model_shelf[model_2]

        scheduler = self_play_parallel.SelfPlayScheduler(policy_container=self.model_shelf[model_1],
                                                         opposing_policy_container=self.model_shelf[model_2],
                                                         env_gen=self.env, epoch_length=num_games, initial_games=0,
                                                         self_play=False)
        _, breakdown = scheduler.compre_models()
        results = {status: breakdown["first"][status] + breakdown["second"][status] for status in
                   ("wins", "draws", "losses")}
        return results

    def calculate_elo(self):
        pass
