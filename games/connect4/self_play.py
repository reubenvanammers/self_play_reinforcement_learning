import copy

import numpy


class SelfPlay:
    # Given a learning policy, opponent policy , learns by playing opponent and then updating opponents model
    # TODO add function for alternating who starts first
    def __init__(self, policy, opposing_policy):
        self.policy = policy
        self.opposing_policy = opposing_policy
        self.alternate_start = False
        self.update_lag = 50  # games till opponent gets updated
        self.q = self.policy.q
        self.env = self.q.env

        self.policy_wins = 0
        self.opponent_wins = 0

    def train_model(self, num_episodes):
        for episode in range(num_episodes):
            if episode % self.update_lag == 0 and episode > 0:
                self.update_opponent_model()
            self.play_episode()

    def play_episode(self):
        s = self.env.reset()
        for i in range(100):  # Should be less than this
            s, done = self.play_round(s)
            if done:
                break

    def swap_state(self, s):
        # Make state as opposing policy will see it
        return s * -1

    def play_round(self, s):
        a = self.policy(s)
        s_next, r, done, info = self.play_move(a, player=1)
        if done:
            self.policy_wins += 1
            self.q.update(s, a, r, done, s_next)
        else:
            s = self.swap_state(s_next)
            a = self.opposing_policy(s)
            s_next, r, done, info = self.play_move(a, player=-1)
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
