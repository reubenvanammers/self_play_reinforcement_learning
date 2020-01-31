from connect4env import Connect4Env, GameOver
import tensorflow as tf
from random import random
import numpy as np
from tensorflow.keras.losses import MeanSquaredError


# Things to do:
# Look at things like gradient tape to speed up learning
# Q learning instead of sarsa
# Use eligibility traces to increase the ability of the system to deal with distant rewards
# Use a convolutional nn to help with spatial representation
# use rollout/monte carlo tree search
# might want to deep copy models if possible?
class Connect4TD:
    def __init__(
            self, epsilon=0.1, self_play_update_freq=10, gamma=1, num_iter=10, width=7, height=6
    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.self_play_update_freq = self_play_update_freq
        self.num_iter = num_iter
        self.width = width
        self.height = height
        self.env = Connect4Env(width=width, height=height)
        self.model = self.build_model()

    def build_model(self):
        # Use afterstates instead of pure sarsa -> essentially a funciton mappng from q(s,a) to w(s') so equivalent to
        # function approximation

        # loss = tf.keras.losses.MeanSquaredError()
        model = tf.keras.models.Sequential(layers=[
            tf.keras.layers.Flatten(
                input_shape=(self.width, self.height)),
            # ),
            tf.keras.layers.Dense(40, activation="relu", input_shape=(self.width, self.height)),
            tf.keras.layers.Dense(1)],
        )
        model.compile(optimizer='SGD', loss=MeanSquaredError())
        return model

    def run(self):
        for iter in range(self.num_iter):
            if iter % 2 == 0:
                first_player = 1
            else:
                first_player = -1
            for episode in range(self.self_play_update_freq):
                self.play_game(first_player=first_player)

    def play_game(self, first_player):
        state, heights = self.env.get_state()
        if first_player == -1:
            enemy_move = self.choose_move(
                state, player=-1
            )
            state, _, game_over = self.play_move(enemy_move)

        move = self.choose_move(state, player=1)
        state, reward, game_over = self.play_move(move, player=1)

        while not game_over:
            try:
                enemy_move = self.choose_move(
                    state, player=-1
                )
                new_state, new_reward, game_over = self.play_move(enemy_move, player=-1)
                move = self.choose_move(state)
                new_state, new_reward, game_over = self.play_move(move, player=1)
                self.update_model(state, new_state, reward)
            except GameOver:
                break

    def update_model(self, state, new_state, reward):
        new_state_value = self.model.predict(np.array([new_state]))
        expected_prev_state_value = reward + self.gamma * new_state_value
        self.model.fit(np.array([state]), expected_prev_state_value)

        def choose_move(self, state, player=1, greedy=False):
            valid_moves = self.env.valid_moves()
            if random() > self.epsilon and not greedy:
                state *= player
                probs = self.model.predict(x=np.array([state]))
                # use an epsilon greedy algoD
                probs = probs * np.array([1 for m in valid_moves if m])
                move = np.argmax(probs)
            else:
                move = np.random.choice([i for i, m in enumerate(valid_moves) if m])

        return move

    def play_move(self, move, player):
        state, reward, game_over = self.env.step(move, player)
        # Multiply reward by player so that if the other player win gives out a negative reward
        return state, reward * player, game_over


if __name__ == "__main__":
    connect = Connect4TD(width=5)
    connect.run()
