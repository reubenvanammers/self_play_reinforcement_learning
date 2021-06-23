import itertools
from collections import namedtuple

import torch

from games.algos import self_play_parallel
from games.algos.model_database import ModelDatabase

result_container = namedtuple("result", ["players", "result"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Elo:
    """
    Class for the calculation of Elo. Requires at least two models in the model database, one of which is set to be
    the anchor class (as Elo is relative).

    elo = Elo(model_database)
    elo.compare_model("random","model_1") # Plays games between these models
    elo.calculate_elo()

    """

    ELO_CONSTANT = 400

    def __init__(self, model_database: ModelDatabase):
        self.model_database = model_database

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
        if key in self.model_database.result_shelf:
            old_results = self.model_database.result_shelf[key]
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
        self.model_database.result_shelf[key] = total_results

    def _get_results(self, model_1, model_2, num_games=100):

        scheduler = self_play_parallel.SelfPlayScheduler(
            policy_container=self.model_database.model_shelf[model_1],
            evaluation_policy_container=self.model_database.model_shelf[model_2],
            env_gen=self.model_database.env,
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
        models = list(self.model_database.model_shelf.keys())

        model_indices = {model: i for i, model in enumerate(model for model in models if model != anchor_model)}
        if "elo" in self.model_database.elo_value_shelf:
            elo_values = self.model_database.elo_value_shelf["elo"]
            initial_weights = [elo_values.get(model, 0) for model in models if model != anchor_model]
            print(initial_weights)
        else:
            initial_weights = None

        self._convert_memory(model_indices)

        model_qs = {model: torch.ones(1, requires_grad=True) for model in models}  # q = 10^(rating/400)
        model_qs[anchor_model] = torch.tensor(10 ** (anchor_elo / self.ELO_CONSTANT), requires_grad=False)
        epoch_length = 1000
        num_epochs = 100
        batch_size = 32
        elo_net = EloNetwork(len(models), initial_weights).to(device)
        optim = torch.optim.SGD([elo_net.elo_vals.weight], lr=400)

        for i in range(num_epochs):
            for j in range(epoch_length):
                optim.zero_grad()

                batch = self.model_database.memory.sample(batch_size)
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
        self.model_database.elo_value_shelf["elo"] = model_elos
        print(model_elos)
        return model_elos

    def _convert_memory(self, model_indices):

        keys = list(self.model_database.result_shelf.keys())
        for key in keys:
            model1, model2 = key.split("__")

            val_1 = self._onehot(model1, model_indices)
            val_2 = self._onehot(model2, model_indices)
            players = torch.stack((val_1, val_2), 1).t()

            results = self.model_database.result_shelf[key]
            result_map = {"wins": 1, "losses": 0, "draws": 0.5}
            for result, value in result_map.items():
                for _ in range(results[result]):
                    self.model_database.memory.add(result_container(players, torch.tensor(value, dtype=torch.float)))

    def _onehot(self, model, model_indices):
        model_idx = model_indices[model] if model in model_indices else None
        if model_idx is not None:
            val = torch.nn.functional.one_hot(torch.tensor(model_idx), len(model_indices))
        else:
            val = torch.zeros(len(model_indices), dtype=torch.long)
        return val


class EloNetwork(torch.nn.Module):
    ELO_CONSTANT = 400

    def __init__(self, num_models, initial_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elo_vals = torch.nn.Linear(num_models - 1, 1)
        with torch.no_grad():
            self.elo_vals.bias.fill_(0.0)
            if initial_weights:
                self.elo_vals.weight.data = torch.FloatTensor(initial_weights)
        self.elo_vals.requires_grad = True

    def forward(self, batch):
        batch = batch.float().to(device)
        batch1, batch2 = torch.split(batch, 1, 1)
        r1 = self.elo_vals.forward(batch1)
        r2 = self.elo_vals.forward(batch2)

        q1 = torch.pow(10, r1 / self.ELO_CONSTANT)
        q2 = torch.pow(10, r2 / self.ELO_CONSTANT)

        expected = q1 / (q1 + q2)
        return expected

    def loss(self, expected, result):
        result_tensor = torch.tensor(result, requires_grad=False, dtype=torch.float).to(device)
        loss = torch.nn.functional.binary_cross_entropy(expected.view(-1), result_tensor)
        return loss
