from rl_utils.memory import Memory


class BaseModel:
    def __init__(self, memory_queue, memory_size, *args, **kwargs):
        self.memory = self.create_memory(memory_size)
        self.create_memory(memory_size)
        self.memory_queue = memory_queue

    def create_memory(self, memory_size):
        return Memory(memory_size)

    def __call__(self, s):
        raise NotImplementedError

    def load_state_dict(self, state_dict, target=False):
        raise NotImplementedError

    def update(self, s, a, r, done, next_s):
        self.push_to_queue(s, a, r, done, next_s)
        self.pull_from_queue()
        if self.ready:
            self.update_from_memory()

    def update_from_memory(self):
        raise NotImplementedError

    @property
    def ready(self):
        raise NotImplementedError

    def state_dict(self):
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


class ModelContainer:
    def __init__(self, policy_gen, policy_args=[], policy_kwargs={}):
        self.policy_gen = policy_gen
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs

    def setup(self, **kwargs):
        if "evaluator" in self.policy_kwargs:
            self.policy_kwargs["network"] = self.policy_kwargs.pop("evaluator")
        return self.policy_gen(*self.policy_args, **self.policy_kwargs, **kwargs)
