import atexit
import itertools
import os
import shelve
from collections import namedtuple

import torch

from games.algos import self_play_parallel
from games.algos.elo import EloNetwork
from games.connect4.connect4env import Connect4Env
from games.general.manual import ManualPlay, View
from rl_utils.memory import Memory

result_container = namedtuple("result", ["players", "result"])


class ModelDatabase:
    """
    This class is used to store the results of different models against each other. Models can be registered and
    persisted, and then given the command to play against eachother. The Elo results can then be calculated,
    assuming that an anchor model (assumed to be random, and set to 0 elo)

    from games.connect4.hardcoded_players import Random
    elo = Elo()
    policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)
    # This persists the model and its arguments to disk
    elo.add_model("model_1", policy_container)

    random_policy = ModelContainer(policy_gen=Random, policy_kwargs={})
    elo.add_model("random", random_policy)

    elo.compare_model("random","model_1") # Plays games between these models
    elo.calculate_elo()

    """

    MODEL_SAVE_FILE = ".ELO_MODEL"
    RESULT_SAVE_FILE = ".ELO_RESULT"
    ELO_VALUE_SAVE_FILE = ".ELO_VALUE"

    ELO_CONSTANT = 400

    def __init__(self, env=Connect4Env):
        path = os.path.dirname(__file__)
        self.model_shelf = shelve.open(os.path.join(path, self.MODEL_SAVE_FILE))
        self.result_shelf = shelve.open(os.path.join(path, self.RESULT_SAVE_FILE))
        self.elo_value_shelf = shelve.open(os.path.join(path, self.ELO_VALUE_SAVE_FILE))

        self.env = env

        self.memory = Memory()

        atexit.register(self._close)

    def _close(self):
        self.model_shelf.close()
        self.result_shelf.close()
        self.elo_value_shelf.close()

    def add_model(self, name, model_container):
        try:
            if self.model_shelf[name]:
                raise ValueError("Model name already in use")
        except KeyError:
            self.model_shelf[name] = model_container
            print(f"added model {name}")

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
            swap = False
        else:
            key = f"{model_2}__{model_1}"
            swap = True
        if key in self.result_shelf:
            old_results = self.result_shelf[key]
        else:
            old_results = {"wins": 0, "draws": 0, "losses": 0}
        new_results = self._get_results(model_1, model_2)
        if swap:
            new_results_ordered = {
                "wins": new_results["losses"],
                "draws": new_results["draws"],
                "losses": new_results["wins"],
            }
        else:
            new_results_ordered = new_results
        total_results = {
            status: new_results_ordered[status] + old_results[status] for status in ("wins", "draws", "losses")
        }
        self.result_shelf[key] = total_results

    def _get_results(self, model_1, model_2, num_games=100):

        scheduler = self_play_parallel.SelfPlayScheduler(
            policy_container=self.model_shelf[model_1],
            opposing_policy_container=self.model_shelf[model_2],
            evaluation_policy_container=self.model_shelf[model_2],
            env_gen=self.env,
            epoch_length=num_games,
            initial_games=0,
            self_play=False,
            save_dir=None,
        )
        _, breakdown = scheduler.compare_models()
        results = {
            status: breakdown["first"][status] + breakdown["second"][status] for status in ("wins", "draws", "losses")
        }
        print(f"{model_1} wins: {results['wins']} {model_2} wins: {results['losses']} draws: {results['draws']}")
        return results

    def calculate_elo(self, anchor_model="random", anchor_elo=0):
        models = list(self.model_shelf.keys())

        model_indices = {model: i for i, model in enumerate(model for model in models if model != anchor_model)}
        if "elo" in self.elo_value_shelf:
            elo_values = self.elo_value_shelf["elo"]
            initial_weights = [elo_values.get(model, 0) for model in models if model != anchor_model]
            print(initial_weights)
        else:
            initial_weights = None

        self._convert_memory(model_indices)

        model_qs = {model: torch.ones(1, requires_grad=True) for model in models}  # q = 10^(rating/400)
        model_qs[anchor_model] = torch.tensor(10 ** (anchor_elo / self.ELO_CONSTANT), requires_grad=False)
        epoch_length = 1000
        num_epochs = 200
        batch_size = 32
        elo_net = EloNetwork(len(models), initial_weights)
        optim = torch.optim.SGD([elo_net.elo_vals.weight], lr=400)

        for i in range(num_epochs):
            for j in range(epoch_length):
                optim.zero_grad()

                batch = self.memory.sample(batch_size)
                batch_t = result_container(*zip(*batch))
                players, results = batch_t
                players = torch.stack(players)
                results = torch.stack(results)
                expected_results = elo_net(players)
                loss = elo_net.loss(expected_results, results)

                loss.backward()
                optim.step()
            for param_group in optim.param_groups:
                param_group["lr"] = param_group["lr"] * 0.99

        model_elos = {
            model: elo_net.elo_vals.weight.tolist()[model_indices[model]] for model in models if model != anchor_model
        }
        model_elos[anchor_model] = anchor_elo
        self.elo_value_shelf["elo"] = model_elos
        print(model_elos)
        return model_elos

    def _convert_memory(self, model_indices):

        keys = list(self.result_shelf.keys())
        for key in keys:
            model1, model2 = key.split("__")

            val_1 = self._onehot(model1, model_indices)
            val_2 = self._onehot(model2, model_indices)
            players = torch.stack((val_1, val_2), 1).t()

            results = self.result_shelf[key]
            result_map = {"wins": 1, "losses": 0, "draws": 0.5}
            for result, value in result_map.items():
                for _ in range(results[result]):
                    self.memory.add(result_container(players, torch.tensor(value, dtype=torch.float)))

    def _onehot(self, model, model_indices):
        model_idx = model_indices[model] if model in model_indices else None
        if model_idx is not None:
            val = torch.nn.functional.one_hot(torch.tensor(model_idx), len(model_indices))
        else:
            val = torch.zeros(len(model_indices), dtype=torch.long)
        return val

    def manual_play(self, model_name):
        model = self.model_shelf[model_name].setup()
        model.train(False)
        model.evaluate(True)

        manual = ManualPlay(Connect4Env(), model)
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
    elo = ModelDatabase()
