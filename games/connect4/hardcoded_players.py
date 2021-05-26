import random
from copy import copy


class OnestepLookahead:
    def __init__(self, env_gen, player=-1, **kwargs):
        self.env = env_gen()
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

    def load_state_dict(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    # determines when a neural net has enough data to train
    @property
    def ready(self):
        pass

    def state_dict(self):
        pass

    def update_target_net(self):
        pass

    def train(self, train_state=False):
        pass

    def evaluate(self, evaluate_state=False):
        pass

    def reset(self, player=None):
        self.player = player
        self.env.reset()

    @property
    def optim(self):
        return None

    def play_action(self, action, player):
        self.env.step(action, player)
        # pass  # does nothign atm - mostly for the mcts


class Random:
    def __init__(self, env_gen, player=-1, **kwargs):
        self.env = env_gen()
        self.env_gen = env_gen
        self.player = player

    def __call__(self, s):
        state, _ = copy(self.env.get_state())
        test_env = self.env_gen()
        possible_moves = [i for i, move in enumerate(self.env.valid_moves()) if move]
        a = random.choice(possible_moves)
        return a

    def load_state_dict(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    # determines when a neural net has enough data to train
    @property
    def ready(self):
        pass

    def state_dict(self):
        pass

    def update_target_net(self):
        pass

    def train(self, train_state=False):
        pass

    def evaluate(self, evaluate_state=False):
        pass

    def reset(self, player=None):
        self.env.reset()

    @property
    def optim(self):
        return None

    def play_action(self, action, player):
        self.env.step(action, player)
