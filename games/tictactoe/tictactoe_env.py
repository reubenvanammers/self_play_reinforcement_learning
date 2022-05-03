from functools import reduce

import numpy as np
from gym import spaces

from games.general.base_env import BaseEnv, GameOver


class TicTacToeEnv(BaseEnv):
    def __init__(self, width=3, height=3, win_amount=3):
        self.height = height
        self.width = width
        self.win_amount = win_amount
        self.episode_over = False
        self.board = np.zeros([width, height], dtype=np.int64)
        self.action_space = spaces.Discrete(width * height)  # Linear action space even though board is 3*3

    def max_moves(self):
        return self.height * self.width

    def step(self, action, player=1):
        if self.episode_over:
            raise GameOver

        loc = self.get_loc(action)
        if not self.board[loc]:
            self.board[loc] = player
        reward = self.get_reward(action, player)
        state = self.board
        self.episode_over = reward != 0 or self.board.all()
        return state, reward, self.episode_over, None
        # Allow invalid moves? Just don't do anything and its a waste?

    def set_state(self, state):
        self.board = state

    def get_loc(self, action):
        return np.unravel_index(action, (self.width, self.height))

    def valid_moves(self):
        return self.board.reshape(-1) == 0

    # Action is a integer between 0 and the width

    def reset(self):
        self.episode_over = False
        self.board = np.zeros([self.width, self.height], dtype=np.int64)
        return self.board

    def render(self, board=None):
        l = []
        board = board or self.board
        map = {0: " ", 1: "X", -1: "O"}
        for row_number in range(self.height):
            l.append("|".join([map[piece] for piece in board[:, row_number]]))
        l.reverse()
        print("\n".join(l))

    def get_reward(self, action, player=1):
        x, y = self.get_loc(action)
        horizontals = self.board[:, y]
        verticals = self.board[x, :]
        offset = y - x
        diagonal_1 = np.diagonal(self.board, offset=offset)
        diagonal_2 = np.diagonal(np.flipud(self.board), offset=y - self.width + x + 1)
        connect = [
            reduce(self._calc_win_in_a_row, row * player, 0) for row in [horizontals, verticals, diagonal_1, diagonal_2]
        ]
        if self.win_amount in connect:
            return 1
        return 0

    def _calc_win_in_a_row(self, x, y):
        if x >= self.win_amount:
            return self.win_amount
        if y > 0:
            # if x > 0:
            return x + y
        return 0

    def get_state(self):
        return self.board, None
