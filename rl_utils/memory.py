from collections import defaultdict, deque

import numpy as np
import torch
import logging

class Memory:
    def __init__(self, max_size=None):
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)
        self.deduplicator = None

    def __len__(self):
        return len(self._buffer)

    def add(self, experience):
        self._buffer.append(experience)
        if self.deduplicator:
            self.deduplicator.add_temp(experience)

    def change_size(self, max_size):
        self.max_size = max_size
        self._buffer = deque(self._buffer, maxlen=max_size)

    def sample(self, batch_size):
        buffer_size = len(self._buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        return [self._buffer[i] for i in index]

    def reset(self):
        self._buffer = deque(maxlen=self.max_size)

    def get_duplicates(self, key):
        length_dict = defaultdict(list)
        keys = torch.stack([getattr(item, key) for item in self._buffer], dim=0)
        unique_keys, inverse_indices = torch.unique(keys, return_inverse=True, dim=0)
        print(f" len of unique keys are {len(unique_keys)}")
        for i, item in enumerate(inverse_indices):
            length_dict[item.item()].append(i)
        # {k: v for k, v in length_dict.items() if len(v) > 1}
        print(f"{len(length_dict)} different entries for {len(self._buffer)} entries")

        return length_dict, unique_keys

    # def deduplicate(self, key, values, named_tuple, maxlen=None):  # is a bit complicated?
    #     # Key: thing in namedtuple to count as what is a duplicate
    #     # Value: field name in named tuples to average over
    #     maxlen = maxlen or self._buffer.maxlen
    #     length_dict, unique_keys = self.get_duplicates(key)
    #     new_buffer = deque(maxlen=maxlen)
    #     for state_index in length_dict:
    #         if len(length_dict[state_index]) == 1:
    #             new_buffer.append(self._buffer[length_dict[state_index][0]])  # Should chckd
    #         else:
    #             tuple_kwarg_dict = {key: unique_keys[state_index]}  # Used for making new namedtuple instance
    #             for v in values:
    #                 # dim = len(getattr(self.buffer[0], v).size())
    #                 tuple_kwarg_dict[v] = torch.mean(
    #                     torch.stack([getattr(self._buffer[i], v) for i in length_dict[state_index]], dim=0, ).float(),
    #                     dim=0,
    #                 )
    #             new_buffer.append(named_tuple(**tuple_kwarg_dict))
    #     self._buffer = new_buffer

    def deduplicate(self, key, values, named_tuple, maxlen=None):
        if not self.deduplicator:
            self.deduplicator = Deduplicator(key=key, values=values, named_tuple=named_tuple, buffer=self._buffer)
        logging.info(f"len of old buffer is {len(self._buffer)}")
        new_buffer = self.deduplicator.deduplicate(max_size=maxlen)
        self._buffer = new_buffer
        logging.info(f"len of new buffer is {len(self._buffer)}")


class Deduplicator:
    def __init__(self, key, values, named_tuple, buffer=None):
        self.key = key
        self.values = values
        self.named_tuple = named_tuple
        self.counter = defaultdict(dict)
        self.temp_queue = deque(buffer) or deque()


    def deduplicate(self, max_size=None):
        for experience in self.temp_queue:
            self.add(experience)
        self.temp_queue = deque()
        return self.create_memory(max_size=max_size)

    def add_temp(self, experience):
        self.temp_queue.append(experience)

    def add(self, experience):
        # Torch tensors aren't hashable
        count = self.counter[getattr(experience, self.key).detach().cpu().numpy().data.tobytes()]
        if count:
            count["count"] += 1
            for value in self.values:
                count[value] += getattr(experience, value)
        else:
            count["count"] = 1
            for value in self.values:
                count[value] = getattr(experience, value)
                count[self.key] = getattr(experience, self.key)

    def create_memory(self, max_size=None):
        buffer = deque(maxlen=max_size)
        for key in self.counter:
            count = self.counter[key]
            tuple_kwarg_dict = {self.key: count[self.key]}  # Used for making new namedtuple instance
            for v in self.values:
                tuple_kwarg_dict[v] = count[v] / count['count']
            buffer.append(self.named_tuple(**tuple_kwarg_dict))
        return buffer
