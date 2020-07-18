from collections import deque, defaultdict
import numpy as np
import torch


class Memory:
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)


    def change_size(self, max_size):
        self.max_size = max_size
        self.buffer = deque(self.buffer,maxlen=max_size)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        return [self.buffer[i] for i in index]

    def reset(self):
        self.buffer = deque(maxlen=self.max_size)

    def get_duplicates(self, key):
        length_dict = defaultdict(list)
        keys = torch.stack([getattr(item, key) for item in self.buffer], dim=0)
        unique_keys, inverse_indices = torch.unique(keys, return_inverse=True, dim=0)
        print(f" len of unique keys are {len(unique_keys)}")
        for i, item in enumerate(inverse_indices):
            length_dict[item.item()].append(i)
        # {k: v for k, v in length_dict.items() if len(v) > 1}
        print(f"{len(length_dict)} different entries for {len(self.buffer)} entries")

        return length_dict, unique_keys

    def deduplicate(self, key, values, named_tuple, maxlen=None):  # is a bit complicated?
        # Key: thing in namedtuple to count as what is a duplicate
        # Value: field name in named tuples to average over
        maxlen = maxlen or self.buffer.maxlen
        length_dict, unique_keys = self.get_duplicates(key)
        new_buffer = deque(maxlen=maxlen)
        for state_index in length_dict:
            if len(length_dict[state_index]) == 1:
                new_buffer.append(self.buffer[state_index])  # Should chckd
            else:
                tuple_kwarg_dict = {key: unique_keys[state_index]}  # Used for making new namedtuple instance
                for v in values:
                    # dim = len(getattr(self.buffer[0], v).size())
                    tuple_kwarg_dict[v] = torch.mean(
                        torch.stack([getattr(self.buffer[i], v) for i in length_dict[state_index]], dim=0,).float(),
                        dim=0,
                    )
                new_buffer.append(named_tuple(**tuple_kwarg_dict))
        self.buffer = new_buffer
