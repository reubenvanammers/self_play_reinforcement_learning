import copy


class GameOver(Exception):
    pass


class BaseEnv:
    def __call__(self):
        return copy.deepcopy(self)

    def num_actions(self):
        return self.action_space.n

    def step(self, action, player=1):
        raise NotImplementedError

    def max_moves(self):
        raise NotImplementedError

    def valid_moves(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def render(self, board=None):
        raise NotImplementedError

    def get_reward(self, action, player=1):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def variant_string(self):
        raise NotImplementedError


class TwoDEnv(BaseEnv):
    def __init__(self, width, height, action_space):
        self.width = width
        self.height = height
        self.action_space = action_space
