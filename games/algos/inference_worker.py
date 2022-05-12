import datetime
import logging
import traceback

import torch
from torch import nn, tensor
from torch.cuda import amp

from games.algos.base_worker import BaseWorker
from games.algos.inference_proxy import device


class InferenceWorker(BaseWorker):
    """
    Worker for batching evaluation results. Pass the inference proxy instead of the network and the actual
    calculations will be batched in the inference worker, processed, and the result sent back.
    This is due to the fact that GPU operations are much more efficient when batched, and can often be slower than
    simply running on the CPU when each worker attempts to evaluate its result by itself.

    """

    def __init__(
        self,
        queues,
        policy: nn.Module,
        evaluation_policy: nn.Module = None,
        epoch_value=None,
        save_dir=None,
        resume=False,
        start_time=None,
    ):
        logging.info("setting up Evaluator worker")
        self.policy = policy.to(device).train(False)
        self.evaluation_policy = evaluation_policy.to(device).train(False) if evaluation_policy else None

        self.policy_queues, self.evaluation_policy_queues = self._expand_queues(queues)

        self.counter = 0
        self.counter_last = 0
        self.counter_diff = 10000
        self.counter_time = datetime.datetime.now()
        self.epoch_value = epoch_value
        self.current_model_file = None
        self.resume = resume

        self.epoch_count = 0

        super().__init__(save_dir=save_dir, start_time=start_time)

    def _expand_queues(self, queues):
        if not queues[0].threaded:
            policy_queues = [queue.policy_queues for queue in queues]
            evaluation_policy_queues = [queue.evaluation_policy_queues for queue in queues]
        else:
            policy_queues = [queue.policy_queues.bidirectional_queues for queue in queues]
            policy_queues = [item for sublist in policy_queues for item in sublist]
            evaluation_policy_queues = [queue.evaluation_policy_queues.bidirectional_queues for queue in queues]
            evaluation_policy_queues = [item for sublist in evaluation_policy_queues for item in sublist]
        return policy_queues, evaluation_policy_queues

    def run(self):
        if self.resume:
            logging.info("loading model into inference proxy")
            self.load_model(prev_run=True)

        while True:
            try:
                if self.epoch_value:
                    if self.epoch_value.value:
                        if self.epoch_value.value != self.epoch_count:
                            logging.info("loading model")
                            self.load_model()
                            self.epoch_count = self.epoch_value.value
                if self.counter > self.counter_last + self.counter_diff:
                    now = datetime.datetime.now()
                    logging.debug(
                        f"Number of requests handled is {self.counter}, {self.counter - self.counter_last} requests took {(now - self.counter_time).total_seconds()} seconds"
                    )
                    self.counter_last = self.counter
                    self.counter_time = now
                self.distribute(self.policy_queues, self.policy)
                if self.evaluation_policy:
                    self.distribute(self.evaluation_policy_queues, self.evaluation_policy)

            except Exception as e:
                traceback.print_exc()
                logging.exception(traceback.format_exc())

    def get_queue_pos(self, queues):
        queue_active = []
        requests = []
        for queue in queues:
            if queue.request_queue.empty():
                queue_active.append(0)
            else:
                requests.append(queue.request_queue.get())
                queue_active.append(1)
        return requests, queue_active

    def distribute(self, queues, evaluator):
        requests, queue_active = self.get_queue_pos(queues)
        if requests:
            policy, value = self.calculate(requests, evaluator)
            j = 0
            i = 0
            while j < len(requests):
                if queue_active[i] == 1:
                    active_queue = queues[i]
                    active_queue.answer_queue.put((policy[j], value[j][0]))
                    j += 1
                i += 1
            self.counter += j

    def calculate(self, requests, evaluator):
        tensor_requests = [tensor(s) for s in requests]
        batch = torch.stack(tensor_requests)
        with amp.autocast():
            policy, value = evaluator.forward(batch)
        return policy.tolist(), value.tolist()
