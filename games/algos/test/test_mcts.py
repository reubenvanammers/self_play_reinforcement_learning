from games.algos.mcts import MCTreeSearch
import numpy as np
import torch
from scipy.special import softmax
from games.tictactoe.tictactoe_env import TicTacToeEnv


class TestEvaluator:

    def __init__(self, actions):
        self.actions = actions
        self.probs = list(softmax(np.arange(actions)/10))
        self.value = 1

    def parameters(self):
        return [torch.tensor(1)]

    def __call__(self, *args, **kwargs):
        return self.probs, self.value


if __name__ == "__main__":
    mcts = MCTreeSearch(TestEvaluator(9), TicTacToeEnv, temperature_cutoff=1)
    state = mcts.reset()
    move = mcts(state)
    print(move)
