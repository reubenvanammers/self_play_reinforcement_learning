from multiprocessing import Queue, Process
from torch import nn, tensor
import torch
from games.algos.base_worker import BaseWorker
from rl_utils.queues import BidirectionalQueue, QueueContainer


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


class EvaluatorWorker(BaseWorker):
    """
    Worker for batching evaluation results and hopefully being more efficient.
    """

    def __init__(
            self,
            queues: list(QueueContainer),
            policy_evaluator: nn.Module,
            opposing_policy_evaluator: nn.Module,
            evaluation_policy_evaluator: nn.Module,
    ):
        self.policy_evaluator = policy_evaluator
        self.opposing_policy_evaluator = opposing_policy_evaluator
        self.evaluation_policy_evaluator = evaluation_policy_evaluator

        self.policy_queues = [queue.policy_queues for queue in queues]
        self.opposing_policy_queues = [queue.opposing_policy_queues for queue in queues]
        self.evaluation_policy_queues = [queue.evaluation_policy_queues for queue in queues]

    def run(self):
        while True:
            self.distribute(self.policy_queues, self.policy_evaluator)

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
        policy, value = self.calculate(requests, evaluator)
        j = 0
        for i in len(queues):
            if queue_active[i]:
                queues[i].answer_queue.put((policy[j], value[j]))
                j += 1
            queue_active.pop()

    def calculate(self, requests, evaluator):
        tensor_requests = [tensor(s) for s in requests]
        batch = torch.stack(tensor_requests)
        policy, value = evaluator.forward(batch)
        return policy.tolist(), value

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
