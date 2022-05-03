import logging
import os
import pickle
from os import listdir
from os.path import isfile, join

import torch
from torch import multiprocessing


class BaseWorker(multiprocessing.Process):
    def __init__(self, start_time, save_dir):
        self.start_time = start_time
        self.save_dir = save_dir

    def load_model(self, prev_run=False, model_file=None):
        logging.info("loading model")
        if model_file:
            recent_file = model_file
        else:
            recent_file = recent_save_file(self.save_dir, self.start_time, prev_run, "model")
        self.current_model_file = recent_file
        self.load_checkpoint(recent_file)

    def load_checkpoint(self, recent_file):
        checkpoint = torch.load(recent_file)

        logging.info(f"loading model from {recent_file}")
        # TODO - deal with target better? Get rid of Q completely?
        self.policy.load_state_dict(checkpoint["model"])

        # if getattr(self, "self_play", None):
        #     self.opposing_policy_train.load_state_dict(checkpoint["model"])

    def load_memory(self, prev_run=False):
        logging.info("loading memory")
        recent_file = recent_save_file(self.save_dir, self.start_time, prev_run, "memory")
        with open(recent_file, "rb") as f:
            self.policy.memory = pickle.load(f)
        self.memory_size = len(self.policy.memory)


def recent_save_file(save_dir, start_time, prev_run=False, starting_str="model"):
    if prev_run:
        folders = [
            join(save_dir, f)
            for f in listdir(os.path.join(save_dir))
            if not isfile(join(save_dir, f)) and f != start_time
        ]
    else:
        folders = [join(save_dir, f) for f in listdir(os.path.join(save_dir)) if not isfile(join(save_dir, f))]
    recent_folder = max(folders)

    saves = [
        join(recent_folder, f)
        for f in listdir(os.path.join(recent_folder))
        if isfile(join(recent_folder, f)) and os.path.split(f)[1].startswith(starting_str)
    ]
    recent_file = max(saves)
    return recent_file
