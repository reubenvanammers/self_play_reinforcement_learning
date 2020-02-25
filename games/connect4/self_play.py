import copy
import math
import numpy
import os
import torch
import datetime

save_dir = 'saves/temp'

from os import listdir
from os.path import isfile, join


class SelfPlay:
    # Given a learning policy, opponent policy , learns by playing opponent and then updating opponents model
    # TODO add function for alternating who starts first
    # TODO add evaluation function
    def __init__(self, policy, opposing_policy, swap_sides=False, benchmark_policy=None):
        self.policy = policy
        self.opposing_policy = opposing_policy
        self.opposing_policy.q.policy_net.train(False)

        if benchmark_policy:
            self.benchmark_policy = benchmark_policy
            self.benchmark_policy.q.policy_net.train(False)

        self.alternate_start = False
        self.update_lag = 500  # games till opponent gets updated
        self.q = self.policy.q
        self.env = self.q.env

        self.policy_wins = 0
        self.opponent_wins = 0
        self.swap_sides = swap_sides

        self.eps_start = 0.3
        self.eps_end = 0.01
        self.eps_decay = 5000

    def evaluate_policy(self, num_episodes):
        episode_list = []
        reward_list = []
        self.policy.q.policy_net.train(False)
        for episode in range(num_episodes):
            s, r = self.play_episode(update=False, swap_sides=episode % 2 == 0)
            episode_list.append(s)
            reward_list.append(r)
            print(f'player {r if r == 1 else 2} won')
        win_percent = sum(1 if r > 0 else 0 for r in reward_list) / len(reward_list) * 100
        print(f"win percent : {win_percent}%")

        self.policy.q.policy_net.train(True)


        return episode_list, reward_list

    def train_model(self, num_episodes, resume=False):

        if resume:
            saves = [f for f in listdir(save_dir) if isfile(join(save_dir, f))]
            recent_file = max(saves)
            self.policy.q.policy_net.load_state_dict(torch.load(join(save_dir, recent_file)))
            self.policy.q.target_net.load_state_dict(torch.load(join(save_dir, recent_file)))
            self.opposing_policy.q.policy_net.load_state_dict(torch.load(join(save_dir, recent_file)))

        for episode in range(num_episodes):
            epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                      math.exp(-1. * episode / self.eps_decay)
            self.policy.epsilon = epsilon

            if episode % self.update_lag == 0 and episode > 0:
                self.update_opponent_model()
            self.play_episode(swap_sides=(self.swap_sides and episode % 2 == 0))
            if episode % 50 == 0:
                print("episode number", episode)
            if episode % 50 == 0 and episode > 0:
                self.update_target_net()
            if episode % 200 == 0 and episode > 0:
                saved_name = os.path.join(save_dir, datetime.datetime.now().isoformat() + ':' + str(episode))
                torch.save(self.policy.q.policy_net.state_dict(), saved_name)

    def play_episode(self, swap_sides=False, update=True):
        s = self.env.reset()
        state_list = []
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)
        for i in range(100):  # Should be less than this
            s, done, r = self.play_round(s, update=update)
            state_list.append(copy.deepcopy(s))
            if done:
                break
        return state_list, r

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
            r = r * player
            return s_next, a, r, done, info

    def play_round(self, s, update=True):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            self.policy_wins += 1
            if update:
                self.q.update(s, own_a, r, done, s_intermediate)
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            if done:
                self.opponent_wins += 1
            if update:
                self.q.update(s, own_a, r, done, s_next)
            return s_next, done, r

    def play_move(self, a, player=1):
        return self.env.step(a, player=player)

    def update_target_net(self):
        print("updating target network")
        self.policy.q.target_net.load_state_dict(self.policy.q.policy_net.state_dict())

    def update_opponent_model(self):
        print("evaluating policy with greedy algo")
        self.policy.epsilon = 0
        self.evaluate_policy(100)
        print("updating policy")
        # self.opposing_policy.q.policy_net.load_state_dict(self.policy.q.policy_net.state_dict())
        # self.policy.q.memory.reset()
        # = copy.deepcopy(
        # self.policy.q
        # )  # See if this works? Might need to use some torch specific stuff
