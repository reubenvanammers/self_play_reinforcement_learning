import datetime
import logging
import os
import time
import traceback
from os.path import join

import multiprocessing_logging
import numpy as np
import torch
from torch import multiprocessing, nn
from torch.utils.tensorboard import SummaryWriter

from games.algos.inference_proxy import InferenceProxy, InferenceWorker
from games.algos.selfplayworker import SelfPlayWorker
from games.algos.updateworker import UpdateWorker
from games.general.base_env import BaseEnv
from games.general.base_model import ModelContainer

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


class SelfPlayScheduler:
    def __init__(
        self,
        policy_container: ModelContainer,
        env_gen: BaseEnv,
        evaluation_policy_container: ModelContainer = None,
        network: nn.Module = None,
        swap_sides=True,
        save_dir="saves",
        epoch_length=500,
        initial_games=64,
        self_play=False,
        lr=0.001,
        stagger=False,
        evaluation_games=100,
        evaluation_network: nn.Module = None,
        stagger_mem_step=5000,
        deduplicate=False,
        update_delay=0.01,
    ):
        self.policy_container = policy_container
        self.evaluation_policy_container = evaluation_policy_container
        self.env_gen = env_gen
        self.swap_sides = swap_sides
        self.save_dir = save_dir
        self.epoch_length = epoch_length
        self.self_play = self_play
        self.lr = lr
        self.stagger = stagger
        self.stagger_mem_step = stagger_mem_step
        self.deduplicate = deduplicate
        self.update_delay = update_delay

        self.network = network
        self.evaluation_games = evaluation_games

        self.start_time = datetime.datetime.now().isoformat()

        self.task_queue = multiprocessing.JoinableQueue()
        self.memory_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.initial_games = initial_games
        self.writer = SummaryWriter()

        self.evaluation_network = evaluation_network

        if save_dir:
            os.mkdir(os.path.join(save_dir, self.start_time))
            logging.basicConfig(filename=join(save_dir, self.start_time, "log"), level=logging.INFO)
        multiprocessing_logging.install_mp_handler()

    # def _get_network(self, container) -> nn.Module:
    #     if container.policy_kwargs.get("evaluator"):
    #         network = container.policy_kwargs["evaluator"]
    #         network.load_state_dict(container.policy_kwargs["starting_state_dict"])
    #         del container.policy_kwargs["evaluator"]
    #     else:
    #         network = None
    #     return network

    def setup_player_workers(self, num_workers=None, inference_proxy=True, threads_per_worker=8, resume_model=False):
        epoch_value = multiprocessing.Value("i", 0)
        num_workers = num_workers or multiprocessing.cpu_count()

        if inference_proxy:
            num_play_workers = num_workers - 2
            assert num_play_workers >= 1
            # Use four worker threads per MCTS game
            queues = [QueueContainer(threading=threads_per_worker * 4) for _ in range(num_play_workers)]
            network_inference_proxy = [InferenceProxy(queue.policy_queues) for queue in queues]

            evaluation_network = self._get_network(self.evaluation_network, self.evaluation_policy_container)
            network = self._get_network(self.network, self.policy_container)

            if evaluation_network:
                evaluation_network_inference_proxy = [
                    InferenceProxy(queue.evaluation_policy_queues) for queue in queues
                ]

            player_workers = [
                SelfPlayWorker(
                    self.task_queue,
                    self.memory_queue,
                    self.result_queue,
                    self.env_gen,
                    network=network_inference_proxy[i],
                    evaluation_network=evaluation_network_inference_proxy[i] if evaluation_network else None,
                    start_time=self.start_time,
                    policy_container=self.policy_container,
                    evaluation_policy_container=self.evaluation_policy_container,
                    save_dir=self.save_dir,
                    self_play=self.self_play,
                    threading=threads_per_worker,
                )
                for i in range(num_play_workers)
            ]

            inference_worker = InferenceWorker(
                queues,
                network,
                epoch_value=epoch_value,
                save_dir=self.save_dir,
                evaluation_policy=evaluation_network,
                resume=resume_model,
                start_time=self.start_time,
            )
            inference_worker.start()
        else:
            num_play_workers = num_workers - 1
            assert num_play_workers >= 1

            network = self.network
            network.share_memory()

            player_workers = [
                SelfPlayWorker(
                    self.task_queue,
                    self.memory_queue,
                    self.result_queue,
                    self.env_gen,
                    network=network,
                    start_time=self.start_time,
                    policy_container=self.policy_container,
                    evaluation_policy_container=self.evaluation_policy_container,
                    save_dir=self.save_dir,
                    resume=resume_model,
                    self_play=self.self_play,
                    epoch_value=epoch_value,
                )
                for _ in range(num_play_workers)
            ]
        return player_workers, inference_worker, epoch_value

    def _get_network(self, network, container) -> nn.Module:
        if network:  # TODO cleanup
            used_network = network
        elif container.policy_kwargs.get("network"):
            used_network = container.policy_kwargs.get("network")
            del container.policy_kwargs["network"]
        elif container.policy_kwargs.get("evaluator"):  # deprecated value
            used_network = container.policy_kwargs.get("evaluator")
            del container.policy_kwargs["evaluator"]
        else:
            used_network = None
        return used_network

    def setup_update_worker(self, resume_memory=False, resume_model=False):

        update_worker_queue = multiprocessing.JoinableQueue()

        update_flag = multiprocessing.Event()
        update_flag.clear()
        network = self.network
        optim = torch.optim.SGD(network.parameters(), weight_decay=0.0001, momentum=0.9, lr=self.lr)

        update_worker = UpdateWorker(
            memory_queue=self.memory_queue,
            policy_container=self.policy_container,
            network=network,
            optim=optim,
            update_flag=update_flag,
            update_worker_queue=update_worker_queue,
            save_dir=self.save_dir,
            resume_memory=resume_memory,
            resume_model=resume_model,
            start_time=self.start_time,
            stagger=self.stagger,
            mem_step=self.stagger_mem_step,
            deduplicate=self.deduplicate,
            update_delay=self.update_delay,
        )
        return update_worker, update_flag, update_worker_queue

    def train_model(
        self,
        num_epochs=10,
        resume_model=False,
        resume_memory=False,
        num_workers=None,
        threads_per_worker=8,
        inference_proxy=True,
    ):
        try:
            player_workers, inference_worker, epoch_value = self.setup_player_workers(
                resume_model=resume_model,
                inference_proxy=inference_proxy,
                threads_per_worker=threads_per_worker,
                num_workers=num_workers,
            )
            [player_worker.start() for player_worker in player_workers]
            update_worker, update_flag, update_worker_queue = self.setup_update_worker(
                resume_memory=resume_memory, resume_model=resume_model
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

            if self.evaluation_games:
                self.evaluate_policy(-1)
            for epoch in range(num_epochs):
                logging.info(f"generating {self.epoch_length} self play games: epoch {epoch}")

                update_flag.set()

                for i in range(self.epoch_length):
                    swap_sides = not i % 2 == 0
                    self.task_queue.put({"play": {"swap_sides": swap_sides, "update": True}})
                self.task_queue.join()
                # Sometimes joinable queue acts a bit weirdly due to full pipes - may be a better solution to this
                time.sleep(5)
                self.task_queue.join()
                time.sleep(5)
                self.task_queue.join()

                logging.info(f"finished generating {self.epoch_length} self play games: epoch {epoch}")

                saved_model_name = os.path.join(
                    self.save_dir,
                    self.start_time,
                    "model-" + datetime.datetime.now().isoformat() + ":" + str(self.epoch_length * (epoch + 1)),
                )
                update_flag.clear()
                update_worker_queue.put({"saved_name": saved_model_name})
                update_worker_queue.join()
                # Update saved name after worker has loaded it
                epoch_value.value += 1
                time.sleep(1)
                # self.task_queue.put({"reference": True})

                reward = self.evaluate_policy(epoch)

                update_worker_queue.put({"reward": reward})

            # Clean up
            update_worker.terminate()
            [w.terminate() for w in player_workers]
            if inference_worker:
                inference_worker.terminate()
            del self.memory_queue
            del self.task_queue
            del self.result_queue
        except Exception as e:
            logging.exception(traceback.format_exc())
            logging.exception("error in main loop" + str(e))

    def run_evaluation_games(self):
        # Requires external validator
        # self.task_queue.put({"reference": True})

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
        logging.info(f"win percent : {win_percent}%")

        logging.info(f"wins: {wins}, draws: {draws}, losses: {losses}")

        starts = ["first", "second"]
        breakdown = {}
        for j, start in enumerate(starts):
            wins = len([i for k, i in enumerate(reward_list) if i["reward"] == 1 and i["swap_sides"] == bool(j)])
            draws = len([i for k, i in enumerate(reward_list) if i["reward"] == 0 and i["swap_sides"] == bool(j)])
            losses = len([i for k, i in enumerate(reward_list) if i["reward"] == -1 and i["swap_sides"] == bool(j)])
            breakdown[start] = dict(wins=wins, draws=draws, losses=losses)
            print(f"starting {start}: wins: {wins}, draws: {draws}, losses: {losses}")
            logging.info(f"starting {start}: wins: {wins}, draws: {draws}, losses: {losses}")

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

    def compare_models(self, num_workers=None, inference_proxy=True, threads_per_worker=8, resume_model=False):
        player_workers, inference_worker = self.setup_player_workers(
            resume_model=resume_model,
            inference_proxy=inference_proxy,
            threads_per_worker=threads_per_worker,
            num_workers=num_workers,
        )

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
        for w in player_workers:
            w.terminate()
        inference_worker.terminate()
        return total_rewards, breakdown
