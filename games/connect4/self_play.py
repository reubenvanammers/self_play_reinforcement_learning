import copy

import numpy


class SelfPlay:
    # Given a learning policy, opponent policy , learns by playing opponent and then updating opponents model
    # TODO add function for alternating who starts first
    def __init__(self, policy, opposing_policy, swap_sides=False):
        self.policy = policy
        self.opposing_policy = opposing_policy
        self.alternate_start = False
        self.update_lag = 50  # games till opponent gets updated
        self.q = self.policy.q
        self.env = self.q.env

        self.policy_wins = 0
        self.opponent_wins = 0
        self.swap_sides = swap_sides

    def train_model(self, num_episodes):
        for episode in range(num_episodes):
            if episode % self.update_lag == 0 and episode > 0:
                self.update_opponent_model()
            self.play_episode(swap_sides=(self.swap_sides and episode % 2 == 0))

    def play_episode(self, swap_sides=False):
        s = self.env.reset()
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)
        for i in range(100):  # Should be less than this
            s, done = self.play_round(s)
            if done:
                break

    def swap_state(self, s):
        # Make state as opposing policy will see it
        return s * -1

    def get_and_play_moves(self, s, player=1):
        if player == 1:
            a = self.policy(s)
            s_next, r, done, info = self.play_move(a, player=1)
            return s_next, a, r, done, info
        else:
            opp_s = self.swap_state(s)
            a = self.opposing_policy(opp_s)
            s_next, r, done, info = self.play_move(a, player=-1)
            return s_next, a, r, done, info

    def play_round(self, s):
        s_next, r, a, done, info = self.get_and_play_moves(s)
        if done:
            self.policy_wins += 1
            self.q.update(s, a, r, done, s_next)
        else:
            s_next, r, a, done, info = self.get_and_play_moves(s_next, player=-1)
            if done:
                self.opponent_wins += 1
            self.q.update(s, a, r, done, s_next)
        return s_next, done

    def play_move(self, a, player=1):
        return self.env.step(a, player=player)

    def update_opponent_model(self):
        self.opposing_policy.q = copy.deepcopy(
            self.policy.q
        )  # See if this works? Might need to use some torch specific stuff
