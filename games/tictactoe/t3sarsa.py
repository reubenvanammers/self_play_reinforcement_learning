import numpy as np
from collections import defaultdict
import itertools
from random import random, choice
import copy


# class keydefaultdict(defaultdict):
#     def __missing__(self, key):
#         if self.default_factory is None:
#             raise KeyError(key)
#         else:
#             ret = self[key] = self.default_factory(key)
#             return ret


class TicTacToeSarsa():

    # Use self play, and epsilon greedy algo on both sides
    # Represent empty space as 0, player 1 as 1 and player 2 as -1 on a n*n array representing the board

    def __init__(self, width=3, epsilon=0.2, self_play_update_freq=50, num_iter=1000):
        self.initial_state = np.zeros([width, width])
        self.width = width
        self.dim = 2
        self.epsilon = epsilon
        self.self_play_update_freq = self_play_update_freq
        self.num_iter = num_iter
        self.alpha = 0.5
        self.gamma = 1
        self.expected_reward = {}  # keydefaultdict(self.default_move_weights)
        self.state_rep_dict = {}

    # def evaluate_reward(self):
    #     pass

    def run(self):
        for iter in range(self.num_iter):
            self.opponent_expected_reward = self.expected_reward
            if iter % 2 == 0:
                first_player = 1
            else:
                first_player = -1
            for episode in range(self.self_play_update_freq):
                self.play_game(first_player=first_player)

    def play_game(self, first_player=1):
        state = self.initial_state
        if not self.expected_reward.get(state.tostring()):
            self.state_rep_dict[state.tostring()] = state
            self.expected_reward[state.tostring()] = self.default_move_weights(state)

        if first_player == -1:
            enemy_move = self.choose_move(state, self.opponent_expected_reward, player=-1)
            state = self.apply_move(state, enemy_move, player=-1)

        if not self.expected_reward.get(state.tostring()):
            self.state_rep_dict[state.tostring()] = state
            self.expected_reward[state.tostring()] = self.default_move_weights(state)

        move = self.choose_move(state, self.expected_reward)

        for _ in range(self.width ** 2):
            reward, new_state = self.play_move(move, state)

            if not self.expected_reward.get(new_state.tostring()):
                self.state_rep_dict[new_state.tostring()] = new_state
                self.expected_reward[new_state.tostring()] = self.default_move_weights(new_state)
            if reward or not self.valid_moves(new_state):
                self.expected_reward[state.tostring()][move] += self.alpha * (
                        reward - self.expected_reward[state.tostring()][move])
                break

            new_move = self.choose_move(new_state, self.expected_reward)
            self.expected_reward[state.tostring()][move] += self.alpha * (
                    reward + self.gamma * self.expected_reward[new_state.tostring()][new_move] -
                    self.expected_reward[state.tostring()][
                        move])
            if reward or not self.valid_moves(new_state):
                break
            state = new_state
            move = new_move

    def play_move(self, move, state):
        state = self.apply_move(state, move, player=1)
        reward = 0
        if winner := self.check_winner(state):
            reward = winner
            return reward, state
        elif winner is None:
            return 0, state

        if not self.opponent_expected_reward.get(state.tostring()):
            self.state_rep_dict[state.tostring()] = state
            self.opponent_expected_reward[state.tostring()] = self.default_move_weights(state)

        enemy_move = self.choose_move(state, self.opponent_expected_reward, player=-1)
        state = self.apply_move(state, enemy_move, player=-1)

        if winner := self.check_winner(state):
            reward = winner
        elif winner is None:
            reward = 0
        return reward, state

    def apply_move(self, state, move, player=1):
        new_state = copy.deepcopy(state)
        new_state[move] = player
        return new_state

    def check_winner(self, state):
        for axis in range(self.dim):
            totals = np.sum(state, axis=axis)
            if self.width in totals:
                return 1
            if -self.width in totals:
                return -1
        diagonals = np.array([np.fliplr(state).diagonal(), state.diagonal()])
        totals = np.sum(diagonals, axis=1)
        if self.width in totals:
            return 1
        if -self.width in totals:
            return -1
        if not self.valid_moves(state):
            return None
        return 0

    def choose_move(self, state, rewards, greedy=False, player=1):
        if random() < self.epsilon and not greedy:
            move = choice(list(rewards[state.tostring()].keys()))
        else:
            player_state = state*player
            if not rewards.get(player_state.tostring()):
                self.state_rep_dict[player_state.tostring()] = player_state
                rewards[player_state.tostring()] = self.default_move_weights(player_state)

            move = max(rewards[player_state.tostring()],
                       key=lambda x: rewards[player_state.tostring()].get(x))  # gets highest valued reward
        return move

    def valid_moves(self, state):
        # state = self.state_rep_dict[state]
        return [(i, j) for (i, j), val in np.ndenumerate(state) if not val]

    def default_move_weights(self, state):
        return dict(zip(self.valid_moves(state), itertools.repeat(0)))


if __name__ == "__main__":
    t = TicTacToeSarsa()
    t.run()
    print(t)
