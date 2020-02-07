from games.connect4.connect4env import Connect4Env
from games.connect4.q import EpsilonGreedy, QLinear
from games.connect4.self_play import SelfPlay

if __name__ == "__main__":
    env = Connect4Env()
    policy = EpsilonGreedy(QLinear(env), 0.1)
    opposing_policy = EpsilonGreedy(QLinear(env), 0)  # Acts greedily
    self_play = SelfPlay(policy, opposing_policy)
    self_play.train_model(20000)
    print("Training Done")
