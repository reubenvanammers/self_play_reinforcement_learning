import ctypes
import datetime
import logging
import os
import traceback
from os.path import join

import multiprocessing_logging
import numpy as np
import torch
from torch import multiprocessing
from torch.utils.tensorboard import SummaryWriter

from games.algos.selfplayworker import SelfPlayWorker
from games.algos.updateworker import UpdateWorker
from games.algos.evaluator_proxy import EvaluatorWorker, EvaluatorProxy

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
from rl_utils.queues import QueueContainer

logging.basicConfig(
    filename="log.log",
    level=logging.INFO,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logging.info("initializing logging")

multiprocessing_logging.install_mp_handler()
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
                self_play=self.self_play,
            )
            for _ in range(num_workers)
        ]
        for w in player_workers:
            w.start()

        for i in range(self.epoch_length):
            swap_sides = not i % 2 == 0
            self.task_queue.put({"play": {"swap_sides": swap_sides, "update": False}, "evaluate": True})
        self.task_queue.join()

        reward_list = []
        while not self.result_queue.empty():
            reward_list.append(self.result_queue.get())

        total_rewards, breakdown = self.parse_results(reward_list)
        return total_rewards, breakdown

    def train_model(self, num_epochs=10, resume_model=False, resume_memory=False, num_workers=None):
        saved_name = multiprocessing.Value('i', 0)
        try:
            num_workers = num_workers or multiprocessing.cpu_count()

            evaluator_proxy = True
            if evaluator_proxy:
                num_play_workers = num_workers - 2
                assert num_play_workers >= 1
                queues = [QueueContainer(threading=5) for _ in range(num_play_workers)]
                evaluators = [EvaluatorProxy(queue.policy_queues) for queue in queues]
                player_workers = [
                    SelfPlayWorker(
                        self.task_queue,
                        self.memory_queue,
                        self.result_queue,
                        self.env_gen,
                        evaluator=evaluators[i],
                        start_time=self.start_time,
                        policy_container=self.policy_container,
                        opposing_policy_container=self.opposing_policy_container,
                        evaluation_policy_container=self.evaluation_policy_container,
                        save_dir=self.save_dir,
                        resume=resume_model,
                        self_play=self.self_play,
                        # model_save_location=saved_name
                    )
                    for i in range(num_play_workers)
                ]
                evaluator_worker = EvaluatorWorker(queues, self.network, model_save_location=saved_name, save_dir=self.save_dir)
                evaluator_worker.start()
            else:
                num_play_workers = num_workers - 1
                assert num_play_workers >= 1

                evaluator = self.network
                evaluator.share_memory()

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
                        model_save_location=saved_name
                    )
                    for _ in range(num_play_workers)
                ]

            for w in player_workers:
                w.start()

            update_worker_queue = multiprocessing.JoinableQueue()

            update_flag = multiprocessing.Event()
            update_flag.clear()
            evaluator = self.network
            optim = torch.optim.SGD(evaluator.parameters(), weight_decay=0.0001, momentum=0.9, lr=self.lr)

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

            logging.info(f"generating {self.initial_games} initial games")
            for i in range(self.initial_games):
                swap_sides = not i % 2 == 0
                self.task_queue.put({"play": {"swap_sides": swap_sides, "update": True}})
            self.task_queue.join()
            logging.info("finished initial evaluation games (if any)")
            while not self.result_queue.empty():
                self.result_queue.get()

            # saved_model_name = None
            reward = self.evaluate_policy(-1)
            for epoch in range(num_epochs):
                logging.info(f"generating {self.epoch_length} self play games: epoch {epoch}")

                update_flag.set()
                for i in range(self.epoch_length):
                    swap_sides = not i % 2 == 0
                    self.task_queue.put(
                        {"play": {"swap_sides": swap_sides, "update": True}}
                    )
                self.task_queue.join()
                logging.info(f"finished generating {self.epoch_length} self play games: epoch {epoch}")

                saved_model_name = os.path.join(
                    self.save_dir,
                    self.start_time,
                    "model-" + datetime.datetime.now().isoformat() + ":" + str(self.epoch_length * epoch),
                )
                update_flag.clear()
                update_worker_queue.put({"saved_name": saved_model_name})
                reward = self.evaluate_policy(epoch)

                update_worker_queue.join()
                # Update saved name after worker has loaded it
                saved_name.value += 1
                update_worker_queue.put({"reward": reward})

            # Clean up
            update_worker.terminate()
            [w.terminate() for w in player_workers]
            if evaluator_proxy:
                evaluator_worker.terminate()
            del self.memory_queue
            del self.task_queue
            del self.result_queue
        except Exception as e:
            logging.exception(traceback.format_exc())
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
            logging.info("running evaluation games")
            print("Running evaluation games")
            self.run_evaluation_games()
            while not self.result_queue.empty():
                reward_list.append(self.result_queue.get())
            print("Evaluation games are: \n")
            logging.info("Evaluation games are ")

            total_rewards, _ = self.parse_results(reward_list)

        # self.scheduler.step(total_rewards)
        print(f"epoch is {epoch}")
        self.writer.add_scalar("total_reward", total_rewards, epoch * self.epoch_length)

        return total_rewards
