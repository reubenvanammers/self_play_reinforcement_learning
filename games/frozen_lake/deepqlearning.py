import tensorflow as tf
import numpy as np
import random
import gym
from collections import deque


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class FrozenLake:
    def __init__(self, memory_size=10000):
        self.env = gym.make("FrozenLake-v0")
        self.model = self.build_model()

        self.action_size = self.env.action_space.n
        self.output_size = self.env.observation_space.n

        self.total_episodes = 15000
        self.gamma = 0.95
        self.decay_rate = 0.005
        self.epsilon = 0.1
        self.max_epsilon = 0.01
        self.min_epsilon = 0.01
        self.max_steps = 16

        self.batch_size = 16
        self.memory = Memory(memory_size)

    def choose_move(self, state, greedy=False):
        # valid_moves = self.env.valid_moves()
        if random.random() > self.epsilon and not greedy:
            probs = self.model.predict(x=np.array([state]))
            # use an epsilon greedy algo
            # probs = probs * np.array([1 for m in valid_moves if m])
            move = np.argmax(probs)
        else:
            move = np.random.choice(range(self.action_size))
            # move = np.random.choice([i for i, m in enumerate(valid_moves) if m])
        return move

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(40, input_shape=(None, self.output_size),
                                  # output_shape=(self.batch_size, self.output_size)
                                  ), tf.keras.layers.Dense(self.action_size)
        ])
        model.compile(loss='MeanSquaredError', optimizer='SGD')
        return model

    def run(self):
        for episode in range(self.total_episodes):
            step = 0
            while step < self.max_steps:
                step += 1

    def pre_populate(self):
        self.env.new_episode()
        for i in range(self.batch_size):
            if i == 0:
                state = self.env.get_state()
            action = np.random.choice(range(self.action_size))
            next_state, reward, done, info = self.env.step(action)
            if done:
                next_state = np.zeros(state.shape)
                self.memory.add((state, action, reward, next_state, done))
                self.env.new_episode()
                state = self.env.get_state()
            else:
                self.memory.add((state, action, reward, next_state, done))
                state = next_state

    def train(self):
        for i in range(self.total_episodes):
            self.env.init()
            state = self.env.init()
            step = 0
            while step < self.max_steps:
                action = self.choose_move(state)
                next_state, reward, done, info = self.env.step(action)
                if done:
                    next_state = np.zeros(state.shape)
                    self.memory.add((state, action, reward, next_state, done))
                else:
                    self.memory.add((state, action, reward, next_state, done))
                    state = next_state

                batch = self.memory.sample(self.batch_size)
                # mini batches
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])
                # self.memory.add()
                qs_next = self.model.predict(next_states_mb)

                qs_target = []

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        qs_target.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + self.gamma * np.max(qs_next[i])
                        qs_target.append(target)
                targets_mb = np.array([each for each in qs_target])

                self.model.fit()

if __name__ == "__main__":
    fl = FrozenLake()
    print('asdf')
