import numpy as np
import itertools
import functools


class JacksCar:
    def __init__(
        self,
        max_cars=20,
        gamma=0.9,
        transfer_cost=2,
        rent_reward=10,
        theta=0.00001,
        max_transfer=5,
        additional_lot_cost=4,
    ):
        self.rent_reward = rent_reward
        self.max_cars = max_cars
        self.gamma = gamma
        self.transfer_cost = transfer_cost

        self.transfer_cost = transfer_cost
        self.request_rate_1 = 3
        self.request_rate_2 = 4
        self.return_rate_1 = 3
        self.return_rate_2 = 2
        self.max_transfer = max_transfer
        self.additional_lot_cost = additional_lot_cost

        self.prob_matrix_1 = self.calculate_prob_matrix(self.request_rate_1, self.return_rate_1)
        self.prob_matrix_2 = self.calculate_prob_matrix(self.request_rate_2, self.return_rate_2)
        self.reward_matrix_1 = self.calculate_reward(self.request_rate_1)
        self.reward_matrix_2 = self.calculate_reward(self.request_rate_2)

        self.theta = theta

        self.value_matrix = self.init_value_matrix()
        self.policy_matrix = self.init_policy_matrix()

    def init_value_matrix(self):
        return np.zeros([self.max_cars + 1, self.max_cars + 1])  # value of cars in first, second lot respectively

    def init_policy_matrix(self):
        return np.zeros([self.max_cars + 1, self.max_cars + 1], dtype=np.int)

    def run(self):
        policy_stable = False
        while not policy_stable:
            self.evaluate_policy_matrix()
            policy_stable = self.improve_policy()
        return self.value_matrix, self.policy_matrix

    def evaluate_policy_matrix(self):
        delta = self.theta + 1
        while delta > self.theta:
            delta = 0
            value_matrix = self.value_matrix
            for num_cars_1, num_cars_2 in itertools.product(range(self.max_cars + 1), repeat=2):
                v = self.value_matrix[num_cars_1, num_cars_2]

                num_cars_moved = self.policy_matrix[num_cars_1, num_cars_2]
                v_s = self.calculate_value(num_cars_1, num_cars_2, num_cars_moved)
                value_matrix[num_cars_1, num_cars_2] = v_s
                delta = max(delta, np.abs(v - v_s))
            self.value_matrix = value_matrix

    def improve_policy(self):

        policy_stable = True
        for num_cars_1, num_cars_2 in itertools.product(range(self.max_cars + 1), repeat=2):
            old_policy = self.policy_matrix[num_cars_1, num_cars_2]
            possible_moves = range(-min(5, num_cars_2), min(5, num_cars_1) + 1)
            values = [(move, self.calculate_value(num_cars_1, num_cars_2, move)) for move in possible_moves]
            best_move = functools.reduce(lambda a, b: a if a[1] > b[1] else b, values)[0]
            self.policy_matrix[num_cars_1, num_cars_2] = best_move
            if old_policy != best_move:
                policy_stable = False
        return policy_stable

    def calculate_value(self, num_cars_1, num_cars_2, num_cars_moved):
        reward = -np.abs(num_cars_moved) * self.transfer_cost  # cars moved from first location to second location
        if num_cars_1 > 10:
            reward -= self.additional_lot_cost
        if num_cars_2 > 10:
            reward -= self.additional_lot_cost
        reward += self.reward_matrix_1[num_cars_1 - num_cars_moved]
        reward += self.reward_matrix_2[num_cars_2 + num_cars_moved]

        value = (
            self.prob_matrix_1[num_cars_1 - num_cars_moved, :]
            .dot(self.value_matrix)
            .dot(self.prob_matrix_2[num_cars_2 + num_cars_moved, :].transpose())
        )
        v_s = reward + self.gamma * value
        return v_s

    def poisson(self, n, rate):
        prob = (rate ** n) * np.exp(-rate) / np.math.factorial(n)
        return prob

    def calculate_prob_matrix(self, request_rate, return_rate):
        max_start = self.max_cars + self.max_transfer
        max_end = self.max_cars

        prob_matrix = np.zeros([max_start + 1, max_end + 1])

        for cars_starting in range(max_start + 1):
            for cars_requested in range(max_start + 1):
                prob_rent = self.poisson(cars_requested, request_rate)
                for cars_returned in range(max_end + 1):
                    prob_returned = self.poisson(cars_returned, return_rate)
                    total_prob = prob_rent * prob_returned
                    end_of_day_cars = max(0, min(cars_starting - cars_requested + cars_returned, 20))
                    prob_matrix[cars_starting, end_of_day_cars] += total_prob
        return prob_matrix

    def calculate_reward(self, request_rate):
        max_start = self.max_cars + self.max_transfer
        expected_reward_matrix = np.zeros([max_start + 1])
        for cars_starting in range(max_start + 1):
            for cars_requested in range(max_start + 1):
                prob_rent = self.poisson(cars_requested, request_rate)
                cars_rented = min(cars_starting, cars_requested)
                money_earned = self.rent_reward * cars_rented

                expected_earnings = prob_rent * money_earned
                expected_reward_matrix[cars_starting] += expected_earnings
        return expected_reward_matrix


if __name__ == "__main__":
    jacks_cars = JacksCar()
    value, policy = jacks_cars.run()
    print(value, policy)
