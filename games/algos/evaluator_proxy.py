from torch import nn, tensor
import torch
from games.algos.base_worker import BaseWorker
from rl_utils.queues import BidirectionalQueue, QueueContainer
import traceback
import logging
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class EvaluatorProxy:
#     """
#     Used for evalutation via queues. Should only be used for evaluation, not for training.
#     """
#
#     def __init__(self, request_queue: Queue, answer_queue: Queue):
#         self.request_queue = request_queue
#         self.answer_queue = answer_queue
#
#     def forward(self, s):
#         self.request_queue.put(s)
#         policy, value = self.answer_queue.get(block=True)
#         return policy, value
#
#     def __call__(self, s, player=1):
#         state = s* player
#         policy, value = self.forward(state)
#         return policy, value * player


class EvaluatorProxy:
    """
    Used for evalutation via queues. Should only be used for evaluation, not for training.
    """

    def __init__(self, queues: BidirectionalQueue):
        self.queues = queues

    def forward(self, s):
        policy, value = self.queues.request(s)
        return policy, value

    def __call__(self, s, player=1):
        state = s * player
        policy, value = self.forward(state)
        return policy, value * player

    # No Op
    def to(self, *args, **kwargs):
        return self

    # train is always false for evaluator proxy
    def train(self, *args, **kwargs):
        return self


class EvaluatorWorker(BaseWorker):
    """
    Worker for batching evaluation results and hopefully being more efficient.

    TODO: Add loading, opposing,evaluation policies
    """

    def __init__(
            self,
            queues,
            policy_evaluator,
            opposing_policy_evaluator=None,
            evaluation_policy_evaluator=None,
    ):
        logging.info("setting up Evaluator worker")
        self.policy_evaluator = policy_evaluator.to(device).train(False)
        # self.opposing_policy_evaluator = opposing_policy_evaluator.to(device).train(False)
        # self.evaluation_policy_evaluator = evaluation_policy_evaluator.to(device).train(False)

        self.policy_queues = [queue.policy_queues for queue in queues]
        self.opposing_policy_queues = [queue.opposing_policy_queues for queue in queues]
        self.evaluation_policy_queues = [queue.evaluation_policy_queues for queue in queues]

        self.counter = 0
        self.counter_last = 0
        self.counter_diff = 10000
        self.counter_time = datetime.datetime.now()

        super().__init__()

    def run(self):
        while True:
            if self.counter > self.counter_last + self.counter_diff:
                now = datetime.datetime.now()
                logging.info(
                    f"Number of requests handled is {self.counter}, {self.counter - self.counter_last} requests took {(now - self.counter_time).total_seconds()} seconds")
                self.counter_last = self.counter
                self.counter_time = now
            try:
                self.distribute(self.policy_queues, self.policy_evaluator)
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
                    if j >= len(policy):
                        print("asdf")
                    active_queue.answer_queue.put((policy[j], value[j][0]))
                    j += 1
                i += 1
                # queue_active.pop()
            self.counter += j

    def calculate(self, requests, evaluator):
        tensor_requests = [tensor(s) for s in requests]
        batch = torch.stack(tensor_requests)
        policy, value = evaluator.forward(batch)
        return policy.tolist(), value.tolist()

    # def get_queue(self, queue):
    #     result_list = [[]]
    #     while not queue.empty():
    #         result_list.append(queue.get())
    #     return result_list

# self.request_queues = request_queues
# self.answer_queues = answer_queues
# self.evaluator = evaluator
#
# self.attatched_workers = len(request_queues)
# assert len(answer_queues) == self.attatched_workers
