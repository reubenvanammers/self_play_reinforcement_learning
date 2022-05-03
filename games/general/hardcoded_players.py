import random
from copy import copy

from games.general.base_env import BaseEnv
from games.general.base_model import BasePlayer


class OneStepLookahead(BasePlayer):
    def __init__(self, env_gen, player=-1, **kwargs):
        self.env: BaseEnv = env_gen()
        self.env_gen = env_gen
        self.player = player

    def __call__(self, s):
        state, _ = copy(self.env.get_state())
        test_env = self.env_gen()
        possible_moves = [i for i, move in enumerate(self.env.valid_moves()) if move]
        for a in possible_moves:  # See if can win from here
            test_env.set_state(copy(state))
            s, r, done, _ = test_env.step(a, self.player)
            if done:
                return a
        for a in possible_moves:  # else see if enemy can win by playing a move - block it
            test_env.set_state(copy(state))
            s, r, done, _ = test_env.step(a, self.player * -1)
            if done:
                return a
        # other wise random
        a = random.choice(possible_moves)
        return a

    def reset(self, player=None):
        self.player = player
        self.env.reset()

    def play_action(self, action, player):
        self.env.step(action, player)


class Random(BasePlayer):
    def __init__(self, env_gen, player=-1, **kwargs):
        self.env = env_gen()
        self.env_gen = env_gen
        self.player = player

    def __call__(self, s):
        state, _ = copy(self.env.get_state())
        possible_moves = [i for i, move in enumerate(self.env.valid_moves()) if move]
        a = random.choice(possible_moves)
        return a

    def reset(self, player=None):
        self.env.reset()

    def play_action(self, action, player):
        self.env.step(action, player)
