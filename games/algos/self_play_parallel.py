import copy
import datetime
import math
import os
from os import listdir
from os.path import isfile, join

# import pickle
from copy import deepcopy
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch import multiprocessing
import pickle
import traceback
import logging
import multiprocessing_logging

logging.basicConfig()
multiprocessing_logging.install_mp_handler()

# save_dir = "saves/temp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    from apex import amp

    if torch.cuda.is_available():
        print("Apex available")
        APEX_AVAILABLE = True
    else:
        APEX_AVAILABLE = False
        print("apex not available")
except ModuleNotFoundError:
    APEX_AVAILABLE = False
    print("apex not available")


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
            initial_games=64,
            self_play=False,
            evaluation_policy_gen=None,
            evaluation_policy_args=[],
            evaluation_policy_kwargs={},
            lr=0.001,
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
        self.self_play = self_play
        self.lr = lr

        self.evaluation_policy_gen = evaluation_policy_gen
        self.evaluation_policy_args = evaluation_policy_args
        self.evaluation_policy_kwargs = evaluation_policy_kwargs
        self.evaluation_games = 100

        self.start_time = datetime.datetime.now().isoformat()
        os.mkdir(os.path.join(save_dir, self.start_time))

        self.task_queue = multiprocessing.JoinableQueue()
        self.memory_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.initial_games = initial_games
        self.writer = SummaryWriter()

        logging.basicConfig(filename=join(save_dir, self.start_time, "log"), level=logging.INFO)
        multiprocessing_logging.install_mp_handler()

        # self.memory = self.policy.memory.get()

    def train_model(self, num_epochs=10, resume_model=False, resume_memory=False, num_workers=None):
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.policy.optim, 'max', patience=10, factor=0.2,
        #                                                             verbose=True)

        evaluator = self.policy_kwargs["evaluator"]  # TODO - fix make nicer
        optim = torch.optim.SGD(evaluator.parameters(), weight_decay=0.0001, momentum=0.9, lr=self.lr)

        # if APEX_AVAILABLE:
        #     opt_level = "O1"
        #     evaluator, optim = amp.initialize(evaluator, optim, opt_level=opt_level)

        self.policy_kwargs["evaluator"] = evaluator

        num_workers = num_workers or multiprocessing.cpu_count()
        player_workers = [
            SelfPlayWorker(
                self.task_queue,
                self.memory_queue,
                self.result_queue,
                self.env_gen,
                start_time=self.start_time,
                policy_gen=self.policy_gen,
                opposing_policy_gen=self.opposing_policy_gen,
                policy_args=deepcopy(self.policy_args),
                policy_kwargs=deepcopy(self.policy_kwargs),
                opposing_policy_args=deepcopy(self.opposing_policy_args),
                opposing_policy_kwargs=deepcopy(self.opposing_policy_kwargs),
                save_dir=self.save_dir,
                resume=resume_model,
                self_play=self.self_play,
                evaluation_policy_gen=self.evaluation_policy_gen,
                evaluation_policy_args=self.evaluation_policy_args,
                evaluation_policy_kwargs=self.evaluation_policy_kwargs,
            )
            for _ in range(num_workers - 1)
        ]
        for w in player_workers:
            w.start()

        policy = self.policy_gen(*self.policy_args, **self.policy_kwargs, memory_queue=self.memory_queue, optim=optim, )
        self.policy_kwargs["optim"] = optim

        save_model_queue = multiprocessing.JoinableQueue()

        update_flag = multiprocessing.Event()
        update_flag.clear()

        update_worker = UpdateWorker(
            self.memory_queue,
            policy_gen=self.policy_gen,
            policy_args=deepcopy(self.policy_args),
            policy_kwargs=deepcopy(self.policy_kwargs),
            update_flag=update_flag,
            save_model_queue=save_model_queue,
            save_dir=self.save_dir,
            resume=resume_memory,
            start_time=self.start_time,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Might need to rework scheduler?
            policy.optim, "max", patience=5, factor=0.2, verbose=True, min_lr=0.00001
        )

        update_worker.start()

        for i in range(self.initial_games):
            swap_sides = not i % 2 == 0
            self.task_queue.put({"play": {"swap_sides": swap_sides, "update": True}})
        self.task_queue.join()
        while not self.result_queue.empty():
            self.result_queue.get()
        # self.result_queue.clearI

        saved_model_name = None
        for epoch in range(num_epochs):
            update_flag.set()
            for i in range(self.epoch_length):
                swap_sides = not i % 2 == 0
                self.task_queue.put(
                    {"play": {"swap_sides": swap_sides, "update": True}, "saved_name": saved_model_name, }
                )
            self.task_queue.join()

            saved_model_name = os.path.join(
                self.save_dir,
                self.start_time,
                "model-" + datetime.datetime.now().isoformat() + ":" + str(self.epoch_length * epoch),
            )
            # torch.save(policy.state_dict(), saved_name)  # also save memory
            # [w.load_model() for w in player_workers]
            update_flag.clear()
            save_model_queue.put(saved_model_name)
            self.evaluate_policy(epoch)
            save_model_queue.join()
            # Do some evaluation?

        update_worker.terminate()
        [w.terminate() for w in player_workers]

    def run_evaluation_games(self):
        for i in range(self.evaluation_games):
            swap_sides = not i % 2 == 0
            self.task_queue.put({"play": {"swap_sides": swap_sides, "update": False}, "evaluate": True})
        self.task_queue.join()

    def parse_results(self, reward_list):
        win_percent = sum(1 if r["reward"] > 0 else 0 for r in reward_list) / len(reward_list) * 100
        wins = len([i["reward"] for i in reward_list if i["reward"] == 1])
        draws = len([i["reward"] for i in reward_list if i["reward"] == 0])
        losses = len([i["reward"] for i in reward_list if i["reward"] == -1])

        print(f"win percent : {win_percent}%")
        print(f"wins: {wins}, draws: {draws}, losses: {losses}")

        starts = ["first", "second"]
        for j, start in enumerate(starts):
            wins = len([i for k, i in enumerate(reward_list) if i["reward"] == 1 and i["swap_sides"] == bool(j)])
            draws = len([i for k, i in enumerate(reward_list) if i["reward"] == 0 and i["swap_sides"] == bool(j)])
            losses = len([i for k, i in enumerate(reward_list) if i["reward"] == -1 and i["swap_sides"] == bool(j)])
            print(f"starting {start}: wins: {wins}, draws: {draws}, losses: {losses}")

        total_rewards = np.sum([r["reward"] for r in reward_list])
        print(f"total rewards are {total_rewards}")
        return total_rewards

    def evaluate_policy(self, epoch):
        logging.info("evaluation policy")
        reward_list = []

        while not self.result_queue.empty():
            reward_list.append(self.result_queue.get())

        total_rewards = self.parse_results(reward_list)

        if self.self_play:
            reward_list = []
            print("Running evaluation games")
            self.run_evaluation_games()
            while not self.result_queue.empty():
                reward_list.append(self.result_queue.get())
            print("Evaluation games are: \n")
            total_rewards = self.parse_results(reward_list)

        self.scheduler.step(total_rewards)
        print(f"epoch is {epoch}")
        self.writer.add_scalar("total_reward", total_rewards, epoch * self.epoch_length)

        return reward_list


class Worker(multiprocessing.Process):
    def load_model(self, prev_run=False):
        logging.info("loading model")

        if prev_run:
            folders = [
                join(self.save_dir, f)
                for f in listdir(os.path.join(self.save_dir))
                if not isfile(join(self.save_dir, f)) and f != self.start_time
            ]
        else:
            folders = [
                join(self.save_dir, f)
                for f in listdir(os.path.join(self.save_dir))
                if not isfile(join(self.save_dir, f))
            ]
        recent_folder = max(folders)

        saves = [
            join(recent_folder, f)
            for f in listdir(os.path.join(recent_folder))
            if isfile(join(recent_folder, f)) and os.path.split(f)[1].startswith("model")
        ]

        recent_file = max(saves)
        self.current_model_file = recent_file
        # self.policy.load_state_dict()
        self.load_checkpoint(recent_file)
        # self.policy.q.target_net.load_state_dict(torch.load(join(self.save_dir, recent_file)))
        # self.opposing_policy.load_state_dict(torch.load(join(self.save_dir, recent_file)))

    def load_checkpoint(self, recent_file):
        checkpoint = torch.load(recent_file)

        self.policy.load_state_dict(checkpoint["model"], target=True)

        if self.self_play:
            self.opposing_policy_train.load_state_dict(checkpoint["model"], target=True)

        if APEX_AVAILABLE:
            amp.load_state_dict(checkpoint["amp"])

    def load_memory(self, prev_run=False):
        logging.info("loading memory")
        if prev_run:
            folders = [
                join(self.save_dir, f)
                for f in listdir(os.path.join(self.save_dir))
                if not isfile(join(self.save_dir, f)) and f != self.start_time
            ]
        else:
            folders = [
                join(self.save_dir, f)
                for f in listdir(os.path.join(self.save_dir))
                if not isfile(join(self.save_dir, f))
            ]
        recent_folder = max(folders)

        saves = [
            join(recent_folder, f)
            for f in listdir(os.path.join(recent_folder))
            if isfile(join(recent_folder, f)) and os.path.split(f)[1].startswith("memory")
        ]
        recent_file = max(saves)
        with open(recent_file, "rb") as f:
            self.policy.memory = pickle.load(f)
        self.memory_size = len(self.policy.memory)


class SelfPlayWorker(Worker):
    def __init__(
            self,
            task_queue,
            memory_queue,
            result_queue,
            env_gen,
            policy_gen,
            opposing_policy_gen,
            start_time,
            policy_args=[],
            policy_kwargs={},
            opposing_policy_args=[],
            opposing_policy_kwargs={},
            save_dir="save_dir",
            resume=False,
            self_play=False,
            evaluation_policy_gen=None,
            evaluation_policy_args=[],
            evaluation_policy_kwargs={},
    ):
        self.env_gen = env_gen
        self.env = env_gen()
        # opposing_policy_kwargs =copy.deepcopy(opposing_policy_kwargs) #TODO make a longer term solutions
        policy_kwargs["memory_queue"] = memory_queue

        self.policy_gen = policy_gen
        self.opposing_policy_gen = opposing_policy_gen
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs
        self.opposing_policy_args = opposing_policy_args
        self.opposing_policy_kwargs = opposing_policy_kwargs
        self.evaluation_policy_gen = evaluation_policy_gen
        self.evaluation_policy_args = evaluation_policy_args
        self.evaluation_policy_kwargs = evaluation_policy_kwargs

        self.task_queue = task_queue
        self.memory_queue = memory_queue
        self.result_queue = result_queue
        self.save_dir = save_dir
        self.start_time = start_time
        self.self_play = self_play

        self.current_model_file = None
        self.resume = resume

        super().__init__()

    def set_up_policies(self):
        self.policy = self.policy_gen(*self.policy_args, **self.policy_kwargs)
        self.policy.train(False)

        self.opposing_policy_train = self.opposing_policy_gen(*self.opposing_policy_args, **self.opposing_policy_kwargs)

        self.opposing_policy_train.train(False)
        # self.opposing_policy.env = self.env
        self.opposing_policy_train.env = self.env_gen()  # TODO: make this a more stabel solution -

        self.opposing_policy_evaluate = (
            self.evaluation_policy_gen(*self.evaluation_policy_args, **self.evaluation_policy_kwargs)
            if self.evaluation_policy_gen
            else None
        )
        if self.opposing_policy_evaluate:
            self.opposing_policy_evaluate.train(False)
            self.opposing_policy_evaluate.env = self.env_gen()  # TODO: make this a more stabel solution -

        self.opposing_policy = self.opposing_policy_train

    def run(self):

        self.set_up_policies()  # dont do in init because of global apex
        if self.resume:
            self.load_model(prev_run=True)

        while True:
            task = self.task_queue.get()
            try:
                if task.get("saved_name") and task.get("saved_name") != self.current_model_file:
                    # time.sleep(5)
                    print("loading model")
                    self.load_model()

                evaluate = task.get("evaluate")
                if evaluate:
                    self.opposing_policy = self.opposing_policy_evaluate
                    self.policy.train(False)
                    self.policy.evaluate(True)
                else:
                    self.opposing_policy = self.opposing_policy_train
                    self.policy.train(False)
                    self.policy.evaluate(False)

                episode_args = task["play"]
                self.play_episode(**episode_args)
                self.task_queue.task_done()
            except Exception as e:
                # traceback.print_exc()
                print(str(e))
                self.task_queue.task_done()

    def play_episode(self, swap_sides=False, update=True):

        s = self.env.reset()
        self.policy.reset(player=(-1 if swap_sides else 1))
        self.opposing_policy.reset(player=(1 if swap_sides else -1))  # opposite from opposing policy perspective
        state_list = []
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)
        for i in range(100):  # Should be less than this
            s, done, r = self.play_round(s, update=update)
            state_list.append(copy.deepcopy(s))
            if done:
                break
        self.result_queue.put({"reward": r, "swap_sides": swap_sides})
        return state_list, r

    def play_round(self, s, update=True):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            if update:
                self.policy.push_to_queue(done, r)
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
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
        self.opposing_policy.play_action(a, player * -1)  # Opposing policy will be player 1, in their perspective
        return self.env.step(a, player=player)


class UpdateWorker(Worker):
    def __init__(
            self,
            memory_queue,
            policy_gen,
            policy_args,
            policy_kwargs,
            update_flag,
            save_model_queue,
            start_time,
            save_dir="saves",
            resume=False,
    ):
        self.memory_queue = memory_queue
        # self.policy = policy
        self.policy_gen = policy_gen
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs
        self.update_flag = update_flag
        self.save_model_queue = save_model_queue
        self.save_dir = save_dir
        self.start_time = start_time
        self.memory_size = 0

        if resume:
            self.load_memory(prev_run=True)
            self.load_model(prev_run=True)

            ##LOAD MODEL STEP!!!!!!!!!!!!! with amp

        super().__init__()

    def run(self):
        self.policy = self.policy_gen(memory_queue=self.memory_queue, *self.policy_args, **self.policy_kwargs)
        self.policy.train()

        while True:
            if not self.save_model_queue.empty():
                saved_name = self.save_model_queue.get()
                self.pull()
                # self.deduplicate_memory()
                self.save_memory()
                self.save_model(saved_name)
                self.save_model_queue.task_done()
            elif self.update_flag.is_set():
                self.update()
            else:
                self.pull()

    def save_model(self, saved_name):
        if APEX_AVAILABLE:
            checkpoint = {
                "model": self.policy.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                "amp": amp.state_dict(),
            }
        else:
            checkpoint = {
                "model": self.policy.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'amp': amp.state_dict()
            }
        torch.save(checkpoint, saved_name)

    def pull(self):
        self.policy.pull_from_queue()
        new_memory_size = len(self.policy.memory)
        if new_memory_size // 1000 - self.memory_size // 1000 > 0:
            self.memory_size = new_memory_size
            self.save_memory()
        self.memory_size = new_memory_size
        time.sleep(1)

    def deduplicate_memory(self):
        print("deduplicating memory")
        self.policy.deduplicate()
        print("deduplication finished")

    def save_memory(self):
        logging.info("saving memory")
        saved_name = os.path.join(
            self.save_dir,
            self.start_time,
            "memory-" + datetime.datetime.now().isoformat() + ":" + str(self.memory_size),
        )
        with open(saved_name, "wb") as f:
            pickle.dump(self.policy.memory, f)

    def update(self):
        # if APEX_AVAILABLE:
        #     amp.load_state_dict(self.policy.state_dict)
        # if not self.save_model_queue.is_set():
        # self.save_model_queue.wait()
        # self.policy.pull_from_queue()
        self.pull()
        self.policy.update_from_memory()
