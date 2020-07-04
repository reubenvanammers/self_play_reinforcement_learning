class BaseModel:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, s):
        raise NotImplementedError

    def load_state_dict(self, state_dict, target=False):
        raise NotImplementedError

    def update(self, s, a, r, done, next_s):
        self.push_to_queue(s, a, r, done, next_s)
        self.pull_from_queue()
        if self.ready:
            self.update_from_memory()

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

    # @property
    # def optim(self):
    #     raise NotImplementedError

    def play_action(self, action, player):
        raise NotImplementedError

    def pull_from_queue(self):
        while not self.memory_queue.empty():
            experience = self.memory_queue.get()
            self.memory.add(experience)

    def push_to_queue(self, s, a, r, done, next_s):
        raise NotImplementedError

    def deduplicate(self):
        pass

    def play_action(self, a):
        raise NotImplementedError
