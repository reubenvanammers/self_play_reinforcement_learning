import torch

from rl_utils.queues import BidirectionalQueue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class InferenceProxy:
    """
    Used for evaluation via queues. Should only be used for inference, not for training.
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

    # train is always false for evaluator proxy
    def load_state_dict(self, *args, **kwargs):
        return self
