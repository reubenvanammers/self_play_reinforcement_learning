import numpy as np
import functools


class CoinFlip:
    def __init__(self, prob_heads=0.5, goal=100, gamma=1):
        self.prob_heads = prob_heads
        self.goal = goal
        self.policy_matrix = np.zeros(goal, dtype=np.int)
        self.value_matrix = np.zeros(goal, dtype=np.longdouble)
        self.gamma = gamma

        self.theta = 0.000000000000000000001

    def run(self):
        self.evaluate_value()
        self.generate_policies()
        return self.value_matrix, self.policy_matrix

    def evaluate_value(self):
        delta = self.theta + 1
        while delta > self.theta:
            delta = 0
            value_matrix = self.value_matrix
            for reserve in range(self.goal):
                v = value_matrix[reserve]
                value = 0
                for stake in range(reserve + 1):
                    stake_value = self.evaluate_stake(reserve, stake)

                    if stake_value >= value:
                        value = stake_value
                self.value_matrix[reserve] = value
                delta = max(delta, np.abs(self.value_matrix[reserve] - v))
            # self.value_matrix = value_matrix

    def evaluate_stake(self, reserve, stake):
        low, high = (reserve - stake, reserve + stake)
        value = (1 - self.prob_heads) * self.value_matrix[low]
        if high >= self.goal:
            value += 1 * self.prob_heads
        else:
            value += self.prob_heads * self.gamma * self.value_matrix[high]
        return value

    def generate_policies(self):
        for reserve in range(1, self.goal):
            values = [(stake, self.evaluate_stake(reserve, stake)) for stake in range(1, reserve + 1)]
            best_stake = functools.reduce(lambda a, b: a if a[1] >= b[1] - self.theta else b, values)[0]
            self.policy_matrix[reserve] = best_stake


if __name__ == "__main__":
    coin = CoinFlip(prob_heads=0.4, goal=128)
    values, policies = coin.run()
    print(values, policies)
