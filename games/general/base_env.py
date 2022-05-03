import copy


class GameOver(Exception):
    pass


class BaseEnv:
    def __call__(self):
        return copy.deepcopy(self)

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
