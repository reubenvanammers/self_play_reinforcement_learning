from multiprocessing import Queue, Process
from torch import nn

from games.algos.base_worker import BaseWorker
from rl_utils.queues import BidirectionalQueue

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
        state = s* player
        policy, value = self.forward(state)
        return policy, value * player


class EvaluatorWorker(BaseWorker):
    """
    Worker for batching evaluation results and hopefully being more efficient.
    """


    def __init__(self, request_queues: list(Queue), answer_queues: list(Queue), evaluator: nn.Module):
        self.request_queues = request_queues
        self.answer_queues = answer_queues
        self.evaluator = evaluator

        self.attatched_workers = len(request_queues)
        assert len(answer_queues) == self.attatched_workers







