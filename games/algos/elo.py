from games.algos import self_play_parallel
from games.connect4.connect4env import Connect4Env
import shelve
import os
import itertools
import atexit
from rl_utils.memory import Memory
from collections import namedtuple
from games.connect4.modules import ConvNetConnect4
from games.algos.mcts import MCTreeSearch
import torch
from games.connect4.hardcoded_players import OnestepLookahead, Random

from games.algos.base_model import ModelContainer

result_container = namedtuple("result", ["players", "result"])


class Elo():
    MODEL_SAVE_FILE = ".ELO_MODEL"
    RESULT_SAVE_FILE = ".ELO_RESULT"

    def __init__(self, env=Connect4Env):
        path = os.path.dirname(__file__)
        self.model_shelf = shelve.open(os.path.join(path, self.MODEL_SAVE_FILE))
        self.result_shelf = shelve.open(os.path.join(path, self.RESULT_SAVE_FILE))

        self.env = env

        self.memory = Memory()

        atexit.register(self._close)

    def _close(self):
        self.model_shelf.close()
        self.result_shelf.close()

    def add_model(self, name, model_container):
        try:
            if self.model_shelf[name]:
                raise ValueError("Model name already in use")
        except KeyError:
            self.model_shelf[name] = model_container
            print("added model {name}")

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
        # model_1_dict = self.model_shelf[model_1]
        # model_2_dict = self.model_shelf[model_2]

        scheduler = self_play_parallel.SelfPlayScheduler(policy_container=self.model_shelf[model_1],
                                                         opposing_policy_container=self.model_shelf[model_2],
                                                         env_gen=self.env, epoch_length=num_games, initial_games=0,
                                                         self_play=False, save_dir=None)
        _, breakdown = scheduler.compare_models()
        results = {status: breakdown["first"][status] + breakdown["second"][status] for status in
                   ("wins", "draws", "losses")}
        print(f"{model_1} wins: {results['wins']} {model_2} wins: {results['losses']} draws: {results['draws']}")
        return results

    def calculate_elo(self, anchor_model="random", anchor_elo=0):
        models = list(self.model_shelf.keys())
        model_indices = {model: i for i, model in enumerate(model for model in models if model != anchor_model)}

        self._convert_memory(model_indices)

        # elos = torch.zeros(len(models) - 1, requires_grad=True)
        model_qs = {model: torch.ones(1, requires_grad=True) for model in models}  # q = 10^(rating/400)
        model_qs[anchor_model] = torch.tensor(10 ** (anchor_elo / 400), requires_grad=False)
        epoch_length = 1000
        num_epochs = 200
        batch_size = 32
        elo_net = EloNetwork(len(models))
        optim = torch.optim.SGD([elo_net.q_vals.weight], lr=800)

        for i in range(num_epochs):
            for j in range(epoch_length):
                optim.zero_grad()

                # result = self.memory.sample(1)[0]
                batch = self.memory.sample(batch_size)
                batch_t = result_container(*zip(*batch))
                players, results = batch_t
                players = torch.stack(players)
                results = torch.stack(results)
                expected_results = elo_net(players)
                loss = elo_net.loss(expected_results, results)

                # loss = self._calculate_loss(model_qs[result.p1], model_qs[result.p2], result.result)
                loss.backward()
                optim.step()
            for param_group in optim.param_groups:
                param_group['lr'] = param_group['lr'] * 0.99

        print(elo_net.q_vals.weight)
        # model_elos = {model: torch.log10(q.data) * 400 for model, q in model_qs.items()}
        model_elos = {model: elo_net.q_vals.weight.tolist()[0][model_indices[model]] for model in models if
                      model != anchor_model}
        model_elos[anchor_model] = anchor_elo
        print(model_elos)
        return model_elos

    def _calculate_loss(self, q1, q2, result):
        expected = q1 / (q1 + q2)
        result_tensor = torch.tensor(result, requires_grad=False, dtype=torch.float)
        loss = torch.nn.functional.l1_loss(expected, result_tensor)
        return loss

    def _convert_memory(self, model_indices):

        keys = list(self.result_shelf.keys())
        for key in keys:
            model1, model2 = key.split("__")

            val_1 = self._onehot(model1, model_indices)
            val_2 = self._onehot(model2, model_indices)
            players = torch.stack((val_1, val_2), 1).t()

            # model_1_idx = model_indices[model1]
            # model_2_idx = model_indices[model2]
            # torch.nn.functional.one_hot(torch.tensor(model_1_idx, model_2_idx), len(model_indices) - 1)

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


class EloNetwork(torch.nn.Module):

    def __init__(self, num_models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_vals = torch.nn.Linear(num_models - 1, 1)  # q = 10^(rating/400)
        with torch.no_grad():
            self.q_vals.bias.fill_(0.)

    def forward(self, batch):
        batch = batch.float()
        batch1, batch2 = torch.split(batch, 1, 1)
        r1 = self.q_vals.forward(batch1)
        r2 = self.q_vals.forward(batch2)

        # r = self.q_vals.forward(batch)
        # q1 = q[0]
        # q2 = q[1]
        # r1, r2 = torch.split(r, 1, 1)
        q1 = torch.pow(10, r1 / 400)
        q2 = torch.pow(10, r2 / 400)

        expected = q1 / (q1 + q2)
        return expected

    def loss(self, expected, result):
        result_tensor = torch.tensor(result, requires_grad=False, dtype=torch.float)
        loss = torch.nn.functional.mse_loss(expected.view(-1), result_tensor)
        return loss


# random, mcts1, onsteplook
if __name__ == "__main__":
    # network = ConvNetConnect4()
    # network.share_memory()

    # policy_gen = MCTreeSearch
    # policy_args = []
    # model_path = "/Users/reuben/PycharmProjects/reinforcement_learning/games/connect4/saves__c4mtcs_par/2020-07-12T15:56:28.076491/model-2020-07-13T05:12:44.321919:2000"
    # model_dict = torch.load(model_path)["model"]
    # policy_kwargs = dict(iterations=400, min_memory=20000, memory_size=20000, env_gen=Connect4Env,
    #                      evaluator=network, starting_state_dict=model_dict
    #                      )
    # policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)
    # opposing_policy_gen = Random
    # opposing_policy_args = []
    # opposing_policy_kwargs = dict(env_gen=Connect4Env)
    # policy_container = ModelContainer(policy_gen=opposing_policy_gen, policy_kwargs=opposing_policy_kwargs)

    elo = Elo()
    # elo.compare_models("mcts1", "onesteplook")

    print(elo.calculate_elo("random", 0))
    # elo.compare_models("random", "mcts1")
    # elo.add_model("mcts1", policy_container)
