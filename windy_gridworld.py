import numpy as np
from random import random, randint, choice


class WindyGridworld:
    def __init__(
        self, use_diagonals=False, allow_stationary=False, stochastic_wind=False
    ):
        self.goal_location = np.array([7, 3])
        self.starting_location = np.array([0, 3])
        self.wind_amounts = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.use_diagonals = use_diagonals
        self.grid_size = (10, 7)
        self.stochastic_wind = stochastic_wind

        self.normal_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.diagonal_actions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        if not self.use_diagonals:
            self.action_list = self.normal_actions
        else:
            self.action_list = self.normal_actions + self.diagonal_actions
        if allow_stationary:
            self.action_list += [(0, 0)]

        self.expected_reward = np.zeros([10, 7, len(self.action_list)])

        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 1

    def run(self, num_episodes):
        for i in range(num_episodes):
            state = self.starting_location
            action = self.choose_action(state)
            for _ in range(100000):
                new_state, reward = self.new_state(state, action)
                q_s_a = self.expected_reward[state[0], state[1], action]
                new_action = self.choose_action(new_state)

                q_s_a_prime = self.expected_reward[
                    new_state[0], new_state[1], new_action
                ]

                self.expected_reward[state[0], state[1], action] += self.alpha * (
                    reward + self.gamma * q_s_a_prime - q_s_a
                )
                state = new_state
                action = new_action
                if np.array_equal(state, self.goal_location):
                    break
        return self.generate_best_path()

    def generate_best_path(self):
        state = self.starting_location
        action = self.choose_action(state, greedy=True)
        action_list = [action]
        state_list = [state]
        for _ in range(1000):
            new_state, _ = self.new_state(state, action)
            new_action = self.choose_action(new_state, greedy=True)
            state = new_state
            action = new_action
            action_list.append(action)
            state_list.append(state)
            if np.array_equal(state, self.goal_location):
                return action_list, state_list

    def generate_wind(self, state):
        x = state[0]
        random_wind = choice([-1, 0, 1])
        return np.array([0, self.wind_amounts[x] + random_wind])

    def new_state(self, state, action):
        action_x_y = np.array(self.action_list[action])
        wind_factor = self.generate_wind(state)
        possible_new_state = state + action_x_y + wind_factor
        new_state = self.impose_boundaries(possible_new_state)
        # if (0 <= possible_new_state[0] < 10 and 0 <= possible_new_state[1] < 7):
        #     new_state = possible_new_state  # going off the grid
        # else:
        #     new_state = state

        if np.array_equal(new_state, self.goal_location):
            reward = 0
        else:
            reward = -1
        return new_state, reward

    def impose_boundaries(self, state):
        for i, boundary in enumerate(self.grid_size):
            if state[i] < 0:
                state[i] = 0
            if state[i] > boundary - 1:
                state[i] = boundary - 1
        return state

    def choose_action(self, state, greedy=False):
        if random() < self.epsilon and not greedy:
            # action = choice(action_list)
            action = randint(0, len(self.action_list) - 1)
        else:
            rewards = self.expected_reward[state[0], state[1], :]
            action = np.argmax(rewards)
        return action


if __name__ == "__main__":
    grid = WindyGridworld(use_diagonals=True, allow_stationary=True)
    actions, states = grid.run(num_episodes=100000)
    actions, states = grid.generate_best_path()
    print(actions, states)
    len_path = len(actions)
    print(len_path)
