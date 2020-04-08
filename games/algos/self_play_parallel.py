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
            initial_games=20000,
    ):
        self.policy_gen = policy_gen
        self.policy_args = policy_args
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
        self.result_queue = multiprocessing.Queue()
        self.initial_games = initial_games
        # self.memory = self.policy.memory.get()

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
                self.result_queue,
                self.env_gen,
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

        policy = self.policy_gen(*self.policy_args, **self.policy_kwargs)

        # while not policy.ready:
        #     if self.task_queue.empty():
        #         for _ in range(10 * num_workers):
        #             self.task_queue.put("play_episode")
        #     time.sleep(1)
        #     policy.pull_from_queue()
        #     # push stuff to task quee
        #     pass
        for i in range(self.initial_games):
            swap_sides = not i % 2 == 0
            self.task_queue.put({"swap_sides": swap_sides, "update": True})
        self.task_queue.join()
        while not self.result_queue.empty():
            self.result_queue.get()
        # self.result_queue.clearI

        update_flag = multiprocessing.Event()
        update_worker = UpdateWorker(self.memory_queue, policy, update_flag)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            policy.optim, "max", patience=30, factor=0.2, verbose=True, min_lr=0.00001
        )

        # for w in update_workers:
        #     w.start()
        update_worker.start()
        for epoch in range(num_epochs):
            update_flag.set()
            for i in range(self.epoch_length):
                swap_sides = not i % 2 == 0
                self.task_queue.put({"swap_sides": swap_sides, "update": True})
            self.task_queue.join()

            saved_name = os.path.join(
                self.save_dir, datetime.datetime.now().isoformat() + ":" + str(self.epoch_length * epoch)
            )
            torch.save(policy.state_dict(), saved_name)  # also save memory
            update_flag.clear()
            self.evaluate_policy(epoch)
            # Do some evaluation?

    def evaluate_policy(self, epoch):
        reward_list = []

        while not self.result_queue.empty():
            reward_list.append(self.result_queue.get())

        #             print(f"player {r if r == 1 else 2} won")
        win_percent = sum(1 if r > 0 else 0 for r in reward_list) / len(reward_list) * 100
        wins = len([i for i in reward_list if i == 1])
        draws = len([i for i in reward_list if i == 0])
        losses = len([i for i in reward_list if i == -1])

        print(f"win percent : {win_percent}%")
        print(f"wins: {wins}, draws: {draws}, losses: {losses}")

        starts = ["first", "second"]
        for j, start in enumerate(starts):
            wins = len(
                [i for k, i in enumerate(reward_list) if i == 1 and (k + 1) % 2 == j]
            )  # k+1 as initial step is going second
            draws = len([i for k, i in enumerate(reward_list) if i == 0 and (k + 1) % 2 == j])
            losses = len([i for k, i in enumerate(reward_list) if i == -1 and (k + 1) % 2 == j])
            print(f"starting {start}: wins: {wins}, draws: {draws}, losses: {losses}")

        total_rewards = np.sum(reward_list)
        print(f"total rewards are {total_rewards}")
        self.scheduler.step(total_rewards)

        writer.add_scalar("total_reward", total_rewards, epoch * self.epoch_length)

        return reward_list


class SelfPlayWorker(multiprocessing.Process):
    def __init__(
            self,
            task_queue,
            memory_queue,
            result_queue,
            env_gen,
            policy_gen,
            opposing_policy_gen,
            policy_args=[],
            policy_kwargs={},
            opposing_policy_args=[],
            opposing_policy_kwargs={},
    ):
        self.env = env_gen()
        policy_kwargs['memory_queue'] = memory_queue
        self.policy = policy_gen(*policy_args, **policy_kwargs)
        self.opposing_policy = opposing_policy_gen(*opposing_policy_args, **opposing_policy_kwargs)
        self.opposing_policy.env = self.env #TODO: make this a more stabel solution -
        self.task_queue = task_queue
        self.memory_queue = memory_queue
        self.result_queue = result_queue
        super().__init__()

    def run(self):
        while True:
            task = self.task_queue.get()

            self.play_episode(**task)
            self.task_queue.task_done()

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
        self.result_queue.put(r)
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
                pass
                # if r == -1:
                # self.opponent_wins += 1
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

    def play_move(self, a, player=1):
        self.policy.play_action(a, player)
        # self.opposing_policy.play_action(a, player)
        return self.env.step(a, player=player)


class UpdateWorker(multiprocessing.Process):
    def __init__(self, memory_queue, policy, update_flag):
        self.memory_queue = memory_queue
        self.policy = policy
        self.update_flag = update_flag

        super().__init__()

    def run(self):
        self.update()

    def update(self):
        while True:
            # if not self.update_flag.is_set():
            self.update_flag.wait()
            self.policy.pull_from_queue()
            self.policy.update_from_memory()
