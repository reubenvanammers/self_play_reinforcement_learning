import copy
import datetime
import math
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch import multiprocessing

# save_dir = "saves/temp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


class SelfPlayScheduler:
    def __init__(
        self,
        policy_gen,
        opposing_policy_gen,
        env_gen,
        policy_args=[],
        policy_kwargs={},
        opposing_policy_args=[],
        opposing_policy_kwargs={},
        swap_sides=True,
        save_dir="saves",
        epoch_length=500,
    ):
        self.policy_gen = policy_gen
        self.policy = policy_gen(*policy_args, **policy_kwargs)
        self.policy_args = (policy_args,)
        self.policy_kwargs = policy_kwargs
        self.opposing_policy_gen = opposing_policy_gen
        self.opposing_policy = opposing_policy_gen(*opposing_policy_args, **opposing_policy_kwargs)
        self.opposing_policy_args = opposing_policy_args
        self.opposing_policy_kwargs = opposing_policy_kwargs
        self.env_gen = env_gen
        self.swap_sides = swap_sides
        self.save_dir = save_dir
        self.epoch_length = epoch_length

        self.task_queue = multiprocessing.JoinableQueue()
        self.memory_queue = multiprocessing.Queue()
        self.memory = self.policy.memory.get()

    def train_model(self, num_epochs=10, resume=False, num_workers=None):
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.policy.optim, 'max', patience=10, factor=0.2,
        #                                                             verbose=True)
        if resume:
            saves = [f for f in listdir(os.path.join(self.save_dir)) if isfile(join(self.save_dir, f))]
            recent_file = max(saves)
            # self.policy.load_state_dict()
            self.policy.load_state_dict(torch.load(join(self.save_dir, recent_file)), target=True)
            # self.policy.q.target_net.load_state_dict(torch.load(join(self.save_dir, recent_file)))
            self.opposing_policy.load_state_dict(torch.load(join(self.save_dir, recent_file)))

        num_workers = num_workers or multiprocessing.cpu_count()
        player_workers = [
            SelfPlayWorker(
                self.task_queue,
                self.memory_queue,
                env_gen=self.env_gen,
                policy_gen=self.policy_gen,
                opposing_policy_gen=self.opposing_policy_gen,
                policy_args=self.policy_args,
                policy_kwargs=self.policy_kwargs,
                opposing_policy_args=self.opposing_policy_args,
                opposing_policy_kwargs=self.opposing_policy_kwargs,
            )
            for _ in range(num_workers - 1)
        ]
        for w in player_workers:
            w.start()

        while not self.policy.ready:
            if self.tasks.empty():
                for _ in range(10 * num_workers):
                    self.tasks.put("play_episode")
                time.sleep(1)
            # push stuff to task quee
            pass

        pause_update = multiprocessing.Event()
        update_workers = [UpdateWorker(self.memory_queue, self.policy, pause_update)]
        for w in update_workers:
            w.start()

        for epoch in range(num_epochs):
            pause_update.unset()
            for _ in range(self.epoch_length):
                self.task_queue.put("play_episode")
            saved_name = os.path.join(
                self.save_dir, datetime.datetime.now().isoformat() + ":" + str(self.epoch_length * epoch)
            )
            torch.save(self.policy.state_dict(), saved_name)  # also save memory
            pause_update.set()
            # Do some evaluation?


class SelfPlayWorker(multiprocessing.Process):
    def __init__(
        self,
        task_queue,
        memory_queue,
        env_gen,
        policy_gen,
        opposing_policy_gen,
        policy_args=[],
        policy_kwargs={},
        opposing_policy_args=[],
        opposing_policy_kwargs={},
    ):
        self.env = env_gen()
        self.policy = policy_gen(*policy_args, **policy_kwargs)
        self.opposing_policy = opposing_policy_gen(*opposing_policy_args, **opposing_policy_kwargs)
        self.task_queue = task_queue
        self.memory_queue = memory_queue

    def play_episode(self, swap_sides=False, update=True):
        s = self.env.reset()
        self.policy.reset(player=-(1 if swap_sides else 1))
        state_list = []
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)
        for i in range(100):  # Should be less than this
            s, done, r = self.play_round(s, update=update)
            state_list.append(copy.deepcopy(s))
            if done:
                break
        return state_list, r

    def play_round(self, s, update=True):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            if r == 1:
                if update:
                    self.policy.push_to_queue(done, r)
                    # self.policy.update(s, own_a, r, done, s_intermediate)
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            if done:
                if r == -1:
                    self.opponent_wins += 1
            if update:
                self.policy.push_to_queue(done, r)
                # self.policy.update(s, own_a, r, done, s_next)
            return s_next, done, r

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


class UpdateWorker(multiprocessing.Process):
    def __init__(self, memory_queue, policy, pause_update):
        self.memory_queue = memory_queue
        self.policy = policy
        self.pause_update = pause_update

    def run(self):
        self.update()

    def update(self):
        while True:
            if not self.pause_update.is_set():
                self.policy.pull_from_queue()
                self.policy.update_from_memory()
