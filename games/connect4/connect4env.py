from functools import reduce

import colorama
import numpy as np
from colorama import Fore, Style
from gym import spaces

from games.general.base_env import BaseEnv, GameOver, TwoDEnv


class Connect4Env(TwoDEnv):
    DEFAULT_WIDTH = 7
    DEFAULT_HEIGHT = 6

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, state=None):
        super().__init__(width, height, spaces.Discrete(width))
        self.episode_over = False  # TODO maybe put this in set state?x

        if state:
            self.set_state(state)
        else:
            self.board = np.zeros([width, height], dtype=np.int64)
            self.heights = np.zeros([width], dtype=np.int64)

    def max_moves(self):
        return self.height * self.width

    # Action is a integer between 0 and the width
    def step(self, action, player=1):
        if self.episode_over:
            raise GameOver
        piece_height = self.heights[action]
        if piece_height < self.height:
            self.board[action, piece_height] = player
            self.heights[action] += 1
        else:
            raise ValueError
        reward = self.get_reward(action, player)
        state = self.board
        self.episode_over = reward != 0 or np.sum(self.heights) == self.height * self.width
        # if self.episode_over:
        #     print('game_over')
        return state, reward, self.episode_over, self.heights

        # Allow invalid moves? Just don't do anything and its a waste?

    def valid_moves(self):
        return self.heights < self.height

    def reset(self):
        self.episode_over = False
        self.board = np.zeros([self.width, self.height], dtype=np.int64)
        self.heights = np.zeros([self.width], dtype=np.int64)
        return self.board

    def set_state(self, state):
        self.board = state
        self.heights = np.sum(np.abs(state), axis=1)

    def render(self, board=None):
        colorama.init()
        l = []
        board = board or self.board
        map = {0: " ", 1: f"{Fore.GREEN}X{Style.RESET_ALL}", -1: f"{Fore.RED}O{Style.RESET_ALL}"}
        for row_number in range(self.height):
            boundary = f"{Fore.BLUE}|{Style.RESET_ALL}"
            l.append(boundary + boundary.join([map[piece] for piece in board[:, row_number]]) + boundary)
        l.reverse()
        l.append(" " + " ".join((str(i) for i in range(self.width))))
        print("\n".join(l))

    def get_reward(self, action, player=1):
        piece_height = self.heights[action] - 1
        horizontals = self.board[:, piece_height]
        verticals = self.board[action, :]
        offset = piece_height - action
        diagonal_1 = np.diagonal(self.board, offset=offset)
        diagonal_2 = np.diagonal(np.flipud(self.board), offset=piece_height - self.width + action + 1)
        connect = [
            reduce(self._calc_win_in_a_row, row * player, 0) for row in [horizontals, verticals, diagonal_1, diagonal_2]
        ]
        if 4 in connect:
            return 1
        return 0

    def _calc_win_in_a_row(self, x, y):
        if x >= 4:
            return 4
        if y > 0:
            # if x > 0:
            return x + y
        return 0

    def get_state(self):
        return self.board, self.heights

    def variant_string(self):
        if self.width == self.DEFAULT_WIDTH and self.height == self.DEFAULT_HEIGHT:
            return "connect4"
        else:
            return f"connect4_{self.width}_{self.height}"
