from games.algos import self_play_parallel
from games.connect4.connect4env import Connect4Env
import shelve
import os
import itertools
import atexit
from rl_utils.memory import Memory
from collections import namedtuple
from games.general.manual import ManualPlay, View
from games.connect4.modules import ConvNetConnect4, DeepConvNetConnect4
from games.algos.mcts import MCTreeSearch
import torch
from games.connect4.hardcoded_players import OnestepLookahead, Random

from games.algos.base_model import ModelContainer

result_container = namedtuple("result", ["players", "result"])


class Elo():
    MODEL_SAVE_FILE = ".ELO_MODEL"
    RESULT_SAVE_FILE = ".ELO_RESULT"
    ELO_VALUE_SAVE_FILE = ".ELO_VALUE"

    ELO_CONSTANT=400

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
            new_results_ordered = {"wins": new_results["losses"], "draws": new_results["draws"],
                                   "losses": new_results["wins"]}
        else:
            new_results_ordered = new_results
        total_results = {status: new_results_ordered[status] + old_results[status] for status in
                         ("wins", "draws", "losses")}
        self.result_shelf[key] = total_results

    def _get_results(self, model_1, model_2, num_games=100):

        scheduler = self_play_parallel.SelfPlayScheduler(policy_container=self.model_shelf[model_1],
                                                         opposing_policy_container=self.model_shelf[model_2],
                                                         evaluation_policy_container=self.model_shelf[model_2],
                                                         env_gen=self.env, epoch_length=num_games, initial_games=0,
                                                         self_play=False, save_dir=None)
        _, breakdown = scheduler.compare_models()
        results = {status: breakdown["first"][status] + breakdown["second"][status] for status in
                   ("wins", "draws", "losses")}
        print(f"{model_1} wins: {results['wins']} {model_2} wins: {results['losses']} draws: {results['draws']}")
        return results

    def calculate_elo_2(self,anchor_model="random", anchor_elo=0):
        k_factor = 5
        models = list(self.model_shelf.keys())

        model_indices = {model: i for i, model in enumerate(model for model in models if model != anchor_model)}
        if "elo" in self.elo_value_shelf:
            elo_values = self.elo_value_shelf["elo"]
            initial_weights = [elo_values.get(model, 0) for model in models if model != anchor_model]
        else:
            initial_weights = None

        self._convert_memory2(model_indices)
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
                param_group['lr'] = param_group['lr'] * 0.99

        model_elos = {model: elo_net.elo_vals.weight.tolist()[model_indices[model]] for model in models if
                      model != anchor_model}
        model_elos[anchor_model] = anchor_elo
        self.elo_value_shelf["elo"] = model_elos
        print(model_elos)
        return model_elos




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
                param_group['lr'] = param_group['lr'] * 0.99

        model_elos = {model: elo_net.elo_vals.weight.tolist()[model_indices[model]] for model in models if
                      model != anchor_model}
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


    def _convert_memory2(self, model_indices):

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
                    if result == "wins":
                        self.memory.add(result_container(players, torch.tensor(value, dtype=torch.float)))
                        self.memory.add(result_container(players, torch.tensor(value, dtype=torch.float)))

                    if result == "losses":
                        self.memory.add(result_container(players, torch.tensor(value, dtype=torch.float)))
                        self.memory.add(result_container(players, torch.tensor(value, dtype=torch.float)))
                    if result == "draws":
                        self.memory.add(result_container(players, torch.tensor(0, dtype=torch.float)))
                        self.memory.add(result_container(players, torch.tensor(1, dtype=torch.float)))


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

        view = View(Connect4Env(), model,opponent_model)
        view.play()


class EloNetwork(torch.nn.Module):
    ELO_CONSTANT=400


    def __init__(self, num_models, initial_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elo_vals = torch.nn.Linear(num_models - 1, 1)
        with torch.no_grad():
            self.elo_vals.bias.fill_(0.)
            if initial_weights:
                self.elo_vals.weight.data = torch.FloatTensor(initial_weights)
        self.elo_vals.requires_grad = True

    def forward(self, batch):
        batch = batch.float()
        batch1, batch2 = torch.split(batch, 1, 1)
        r1 = self.elo_vals.forward(batch1)
        r2 = self.elo_vals.forward(batch2)

        q1 = torch.pow(10, r1 / self.ELO_CONSTANT)
        q2 = torch.pow(10, r2 / self.ELO_CONSTANT)

        expected = q1 / (q1 + q2)
        return expected

    # def loss(self, expected, result):
    #     result_tensor = torch.tensor(result, requires_grad=False, dtype=torch.float)
    #     loss = torch.nn.functional.l1_loss(expected.view(-1), result_tensor)
    #     return loss

    def loss(self, expected, result):
        result_tensor = torch.tensor(result, requires_grad=False, dtype=torch.float)
        loss = torch.nn.functional.binary_cross_entropy(expected.view(-1), result_tensor)
        return loss

    # random, mcts1, onsteplook cloudmcts (model-2020-05-13T22_26_13.689443_4000), cloudmcts2 (model-2020-05-01T00_01_10.394613_2500), 15layer-num1
if __name__ == "__main__":
    elo = Elo()
    elo.observe("cloudmcts", "15layer-num1")
    # elo.compare_models("15layer-num1", "onesteplook")

    # print(elo.calculate_elo("random", 0))
    # elo._close()

    #     network = DeepConvNetConnect4()
    #     network.share_memory()
    #     #
    #     policy_gen = MCTreeSearch
    #     policy_args = []
    #     model_path = "/Users/reuben/PycharmProjects/reinforcement_learning/15layermodel_0"
    #     model_dict = torch.load(model_path,map_location=torch.device('cpu'))["model"]
    #     policy_kwargs = dict(iterations=400, min_memory=20000, memory_size=20000, env_gen=Connect4Env,
    #                          evaluator=network, starting_state_dict=model_dict
    #                          )
    #     policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)
    #     opposing_policy_gen = Random
    #     opposing_policy_args = []
    #     opposing_policy_kwargs = dict(env_gen=Connect4Env)
    #     policy_container = ModelContainer(policy_gen=opposing_policy_gen, policy_kwargs=opposing_policy_kwargs)

    # elo = Elo()
    # elo.manual_play("15layer-num1")
    # elo.add_model("15layer-num1", policy_container)

    # elo.compare_models("15layer-num1", "mcts1")
    # elo.compare_models("cloudmcts", "cloudmcts2")

    # print(elo.calculate_elo("random", 0))
    # elo._close()
    # elo.add_model("cloudmcts", policy_container)
    # elo.compare_models("onesteplook", "cloudmcts")
