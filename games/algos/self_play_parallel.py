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

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    filename="log.log",
    level=logging.INFO,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logging.info("initializing logging")

multiprocessing_logging.install_mp_handler()
logging.info("initializing logging2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    from apex import amp

    if torch.cuda.is_available():
        print("Apex available")
        APEX_AVAILABLE = False
    else:
        APEX_AVAILABLE = False
        print("apex not available")
except ModuleNotFoundError:
    APEX_AVAILABLE = False
    print("apex not available")
try:
    from torch2trt import TRTModule

    TRT_AVAILABLE = True
except ModuleNotFoundError:
    TRT_AVAILABLE = False


class SelfPlayScheduler:
    def __init__(
            self,
            policy_container,
            opposing_policy_container,
            env_gen,
            evaluation_policy_container=None,
            network=None,
            swap_sides=True,
            save_dir="saves",
            epoch_length=500,
            initial_games=64,
            self_play=False,
            lr=0.001,
            stagger=False,
            evaluation_games=100,
    ):
        self.policy_container = policy_container
        self.opposing_policy_container = opposing_policy_container
        self.evaluation_policy_container = evaluation_policy_container
        self.env_gen = env_gen
        self.swap_sides = swap_sides
        self.save_dir = save_dir
        self.epoch_length = epoch_length
        self.self_play = self_play
        self.lr = lr
        self.stagger = stagger

        self.network = network
        self.evaluation_games = evaluation_games

        self.start_time = datetime.datetime.now().isoformat()


        self.task_queue = multiprocessing.JoinableQueue()
        self.memory_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.initial_games = initial_games
        self.writer = SummaryWriter()

        if save_dir:
            os.mkdir(os.path.join(save_dir, self.start_time))
            logging.basicConfig(filename=join(save_dir, self.start_time, "log"), level=logging.INFO)
        multiprocessing_logging.install_mp_handler()

        # self.memory = self.policy.memory.get()

    def compare_models(self, num_workers=None):
        num_workers = num_workers or multiprocessing.cpu_count()
        player_workers = [
            SelfPlayWorker(
                self.task_queue,
                self.memory_queue,
                self.result_queue,
                self.env_gen,
                # evaluator=evaluator,
                start_time=self.start_time,
                policy_container=self.policy_container,
                opposing_policy_container=self.opposing_policy_container,
                evaluation_policy_container=self.evaluation_policy_container,
                save_dir=self.save_dir,
                # resume=resume_model,
                self_play=self.self_play,
            ) for _ in range(num_workers)]
        for w in player_workers:
            w.start()

        for i in range(self.epoch_length):
            swap_sides = not i % 2 == 0
            self.task_queue.put(
                {"play": {"swap_sides": swap_sides, "update": False}, "evaluate": True}
            )
        self.task_queue.join()

        reward_list = []
        while not self.result_queue.empty():
            reward_list.append(self.result_queue.get())

        total_rewards, breakdown = self.parse_results(reward_list)
        return total_rewards, breakdown

    def train_model(self, num_epochs=10, resume_model=False, resume_memory=False, num_workers=None):
        try:
            evaluator = self.network
            optim = torch.optim.SGD(evaluator.parameters(), weight_decay=0.0001, momentum=0.9, lr=self.lr)
            evaluator.share_memory()

            num_workers = num_workers or multiprocessing.cpu_count()
            player_workers = [
                SelfPlayWorker(
                    self.task_queue,
                    self.memory_queue,
                    self.result_queue,
                    self.env_gen,
                    evaluator=evaluator,
                    start_time=self.start_time,
                    policy_container=self.policy_container,
                    opposing_policy_container=self.opposing_policy_container,
                    evaluation_policy_container=self.evaluation_policy_container,
                    save_dir=self.save_dir,
                    resume=resume_model,
                    self_play=self.self_play,
                )
                for _ in range(num_workers - 1)
            ]
            for w in player_workers:
                w.start()

            update_worker_queue = multiprocessing.JoinableQueue()

            update_flag = multiprocessing.Event()
            update_flag.clear()

            update_worker = UpdateWorker(
                memory_queue=self.memory_queue,
                policy_container=self.policy_container,
                evaluator=evaluator,
                optim=optim,
                update_flag=update_flag,
                update_worker_queue=update_worker_queue,
                save_dir=self.save_dir,
                resume=resume_memory,
                start_time=self.start_time,
                stagger=self.stagger,
            )

            update_worker.start()

            for i in range(self.initial_games):
                swap_sides = not i % 2 == 0
                self.task_queue.put({"play": {"swap_sides": swap_sides, "update": False}})
            self.task_queue.join()
            while not self.result_queue.empty():
                self.result_queue.get()

            saved_model_name = None
            reward = self.evaluate_policy(-1)
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
                update_flag.clear()
                update_worker_queue.put({"saved_name": saved_model_name})
                reward = self.evaluate_policy(epoch)

                update_worker_queue.join()
                update_worker_queue.put({"reward": reward})

                # Do some evaluation?

            # Clean up
            update_worker.terminate()
            [w.terminate() for w in player_workers]
            del self.memory_queue
            del self.task_queue
            del self.result_queue
        except Exception as e:
            logging.exception("error in main loop" + str(e))

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
        logging.info(f"wins: {wins}, draws: {draws}, losses: {losses}")

        starts = ["first", "second"]
        breakdown = {}
        for j, start in enumerate(starts):
            wins = len([i for k, i in enumerate(reward_list) if i["reward"] == 1 and i["swap_sides"] == bool(j)])
            draws = len([i for k, i in enumerate(reward_list) if i["reward"] == 0 and i["swap_sides"] == bool(j)])
            losses = len([i for k, i in enumerate(reward_list) if i["reward"] == -1 and i["swap_sides"] == bool(j)])
            breakdown[start] = dict(wins=wins, draws=draws, losses=losses)
            print(f"starting {start}: wins: {wins}, draws: {draws}, losses: {losses}")

        total_rewards = np.sum([r["reward"] for r in reward_list])
        logging.info(f"rewards are {total_rewards}")
        print(f"total rewards are {total_rewards}")
        return total_rewards, breakdown

    def evaluate_policy(self, epoch):
        logging.info("evaluation policy")
        reward_list = []
        while not self.result_queue.empty():
            reward_list.append(self.result_queue.get())
        if epoch >= 0:
            total_rewards, _ = self.parse_results(reward_list)

        if self.self_play:
            reward_list = []
            print("Running evaluation games")
            self.run_evaluation_games()
            while not self.result_queue.empty():
                reward_list.append(self.result_queue.get())
            print("Evaluation games are: \n")
            total_rewards, _ = self.parse_results(reward_list)

        # self.scheduler.step(total_rewards)
        print(f"epoch is {epoch}")
        self.writer.add_scalar("total_reward", total_rewards, epoch * self.epoch_length)

        return total_rewards


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

        if getattr(self, "self_play", None):
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
            policy_container,
            opposing_policy_container,
            start_time,
            evaluator=None,
            save_dir="save_dir",
            resume=False,
            self_play=False,
            evaluation_policy_container=None,

    ):
        logging.info("initializing worker")
        self.env_gen = env_gen
        self.env = env_gen()
        self.evaluator = evaluator
        # opposing_policy_kwargs =copy.deepcopy(opposing_policy_kwargs) #TODO make a longer term solutions
        # policy_kwargs["memory_queue"] = memory_queue

        self.policy_container = policy_container
        self.opposing_policy_container = opposing_policy_container
        self.evaluation_policy_container = evaluation_policy_container

        self.opposing_policy_evaluate = None

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
        try:
            # self.policy = self.policy_gen(
            #     memory_queue=self.memory_queue, evaluator=self.evaluator, *self.policy_args, **self.policy_kwargs
            # )

            if self.self_play:
                self.policy = self.policy_container.setup(memory_queue=self.memory_queue, evaluator=self.evaluator)
                self.opposing_policy_train = self.opposing_policy_container.setup(evaluator=self.evaluator)
                # self.opposing_policy_train = self.opposing_policy_gen(
                #     evaluator=self.evaluator, *self.opposing_policy_args, **self.opposing_policy_kwargs
                # )
            else:
                self.policy = self.policy_container.setup(memory_queue=self.memory_queue)
                self.opposing_policy_train = self.opposing_policy_container.setup()

            self.opposing_policy_train.train(False)
            self.policy.train(False)

            self.opposing_policy_train.env = self.env_gen()  # TODO: make this a more stabel solution -

            if self.evaluation_policy_container:
                self.opposing_policy_evaluate = self.evaluation_policy_container.setup()

            if self.opposing_policy_evaluate:
                self.opposing_policy_evaluate.evaluate(True)
                self.opposing_policy_evaluate.train(False)
                self.opposing_policy_evaluate.env = self.env_gen()  # TODO: make this a more stabel solution -

            self.opposing_policy = self.opposing_policy_train
            logging.info("finished setting up policies")
        except Exception as e:
            logging.exception("setting up policies failed"+str(e))

    def run(self):
        logging.info("running self play worker")

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
                logging.info("task done")
            except Exception as e:
                raise e
                # traceback.print_exc()
                # print(str(e))
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
                self.policy.push_to_queue(s, own_a, r, done, s_intermediate)
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            if update:
                self.policy.push_to_queue(s, own_a, r, done, s_next)
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
            evaluator,
            optim,
            policy_container,
            update_flag,
            update_worker_queue,
            start_time,
            save_dir="saves",
            resume=False,
            stagger=False,
    ):
        logging.info("initializing update worker")
        self.memory_queue = memory_queue
        self.evaluator = evaluator
        self.optim = optim
        self.policy_container = policy_container
        self.update_flag = update_flag
        self.update_worker_queue = update_worker_queue
        self.save_dir = save_dir
        self.start_time = start_time
        self.memory_size = 0
        self.resume = resume
        self.stagger = stagger

        self.mem_step = 2500
        self.max_mem = 100000

        ##LOAD MODEL STEP!!!!!!!!!!!!! with amp

        super().__init__()

    def run(self):
        logging.info("running update worker")
        try:
            self.policy = self.policy_container.setup(memory_queue=self.memory_queue, evaluator=self.evaluator,
                                                    optim=self.optim)
            self.policy.train()

            if self.resume:
                self.load_memory(prev_run=True)
                self.load_model(prev_run=True)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Might need to rework scheduler?
                self.policy.optim, "max", patience=10, factor=0.2, verbose=True, min_lr=0.00001
            )

            while True:
                if not self.update_worker_queue.empty():
                    task = self.update_worker_queue.get()
                    if task.get("saved_name"):
                        saved_name = task["saved_name"]
                        self.pull()
                        # self.deduplicate_memory()
                        if self.stagger:
                            self.stagger_memory()
                        self.save_memory()
                        self.save_model(saved_name)
                    elif task.get("reward"):
                        reward = task.get("reward")
                        self.scheduler.step(reward)
                    self.update_worker_queue.task_done()
                elif self.update_flag.is_set():
                    self.update()
                else:
                    self.pull()
        except Exception as e:
            logging.exception("error in update worker" +str(e))

    def stagger_memory(self):
        max_size = min(self.policy.memory.max_size + self.mem_step, self.max_mem)
        self.policy.memory.change_size(max_size)

    def save_model(self, saved_name):
        logging.info("saving model")
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
        #time.sleep(1)

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
        self.pull()
        logging.info("updating from memory")
        for _ in range(100):
            time.sleep(0.10)
            self.policy.update_from_memory()
