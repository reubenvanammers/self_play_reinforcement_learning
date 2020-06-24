

class BaseModel:
    def __init__(self, *args, **kwargs) :
        pass

    def __call__(self, s):
        raise NotImplementedError

    def load_state_dict(self, state_dict, target=False):
        raise NotImplementedError


    def update(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def ready(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def update_target_net(self):
        raise NotImplementedError

    def train(self, train_state):
        raise NotImplementedError

    def evaluate(self, evaluate_state=False):
        pass

    def reset(self):
        raise NotImplementedError

    @property
    def optim(self):
        raise NotImplementedError

    def play_action(self, action, player):
        raise NotImplementedError

    def pull_from_queue(self):
        raise NotImplementedError

    def push_to_queue(self, done, r):
        raise NotImplementedError

    def deduplicate(self):
        pass
