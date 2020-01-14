import numpy as np
from functools import reduce


class Connect4Env:

    def __init__(self, width=7, height=6):
        self.height = height
        self.width = width
        self.board = np.zeros([width, height])
        self.heights = np.zeros([width])

    def step(self, action, player=1):
        piece_height = self.heights[action] + 1
        if piece_height < self.height:
            self.board[action, piece_height] = player
            self.heights[action] += 1
        reward = self.get_reward(action)
        state = self.board
        episode_over = reward != 0

        return state, reward, episode_over

        # Allow invalid moves? Just don't do anything and its a waste?

    # Action is a integer between 0 and the width

    def reset(self, action):
        self.board = np.zeros([self.width, self.height])
        self.heights = np.zeros([self.width])

    def get_reward(self, action, player=1):
        piece_height = self.heights[action]
        horizontals = self.board[:, piece_height]
        verticals = self.board[action, :]
        offset = piece_height - action
        diagonal_1 = np.diagonal(self.board, offset=offset)
        diagonal_2 = np.diagonal(np.fliplr(self.board), offset=piece_height - self.height + action + 1)
        connect = [reduce(self._calc_win_in_a_row, row * player) for row in
                   [horizontals, verticals, diagonal_1, diagonal_2]]
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
