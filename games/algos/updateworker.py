import datetime
import logging
import os
import pickle
import time
import traceback

import torch

from games.algos.base_worker import BaseWorker


class UpdateWorker(BaseWorker):
    def __init__(
        self,
        memory_queue,
        network,
        optim,
        policy_container,
        update_flag,
        update_worker_queue,
        start_time,
        save_dir="saves",
        resume=False,
        stagger=False,
        mem_step=5000,
        max_mem=500000,
        deduplicate=False,
    ):
        logging.info("initializing update worker")
        self.memory_queue = memory_queue
        self.network = network
        self.optim = optim
        self.policy_container = policy_container
        self.update_flag = update_flag
        self.update_worker_queue = update_worker_queue
        self.save_dir = save_dir
        self.start_time = start_time
        self.memory_size = 0
        self.resume = resume
        self.stagger = stagger
        self.deduplicate = deduplicate

        self.mem_step = mem_step
        self.max_mem = max_mem

        self.recent_save = None

        self.memory_size_step = 50000

        super().__init__()

    def run(self):
        logging.info("running update worker")
        try:
            self.policy = self.policy_container.setup(
                memory_queue=self.memory_queue, network=self.network, optim=self.optim,
            )
            self.policy.train()

            if self.resume:
                self.load_memory(prev_run=True)
                self.load_model(prev_run=True)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Might need to rework scheduler?
                self.policy.optim, "max", patience=20, factor=0.5, verbose=True, min_lr=0.00001,
            )

            while True:
                if not self.update_worker_queue.empty():
                    task = self.update_worker_queue.get()
                    if task.get("saved_name"):
                        saved_name = task["saved_name"]
                        self.pull()
                        if self.stagger:
                            self.stagger_memory()
                        if self.deduplicate:
                            self.policy.deduplicate()
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
            logging.exception(traceback.format_exc())
            logging.exception("error in update worker" + str(e))

    def stagger_memory(self):
        max_size = min(self.policy.memory.max_size + self.mem_step, self.max_mem)
        self.policy.memory.change_size(max_size)

    def save_model(self, saved_name):
        logging.info("saving model")
        checkpoint = {
            "model": self.policy.state_dict(),
        }

        torch.save(checkpoint, saved_name)

    def pull(self):
        self.policy.pull_from_queue()
        new_memory_size = len(self.policy.memory)
        if new_memory_size // self.memory_size_step - self.memory_size // self.memory_size_step > 0:
            self.memory_size = new_memory_size
            self.save_memory()
        self.memory_size = new_memory_size

    def save_memory(self):
        logging.info("saving memory")
        saved_name = os.path.join(
            self.save_dir,
            self.start_time,
            "memory-" + datetime.datetime.now().isoformat() + ":" + str(self.memory_size),
        )
        with open(saved_name, "wb") as f:
            pickle.dump(self.policy.memory, f)
        if self.recent_save:
            logging.info(f"removing {self.recent_save}")
            os.remove(self.recent_save)
        self.recent_save = saved_name

    def update(self):
        self.pull()
        logging.debug("updating from memory")
        for _ in range(100):
            # We are creating new games at the same time we update our model. This is more limited by the running
            # of new games, so we rate limit the updates to speed up the evaluation and help prevent overfitting
            time.sleep(0.01)
            self.policy.update_from_memory()
