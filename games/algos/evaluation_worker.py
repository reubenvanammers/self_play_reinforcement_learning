import logging

from c4_perfect_player.connect4_perfect_player import PerfectPlayer
import numpy as np
import csv
# p = PerfectPlayer(book_dir='/home/reuben/projects/update/connect4/7x6.book')
#
# state = p.get_one_position_score(np.array([3,3]))
from games.algos.base_model import ModelContainer, BaseModel
from games.algos.mcts import MCTreeSearch
from games.connect4.connect4env import Connect4Env
from games.connect4.modules import DeepConvNetConnect4
import random


class PerfectEvaluator:

    def __init__(self, evaluator: MCTreeSearch):
        self.perfect_player = PerfectPlayer(book_dir='/home/reuben/projects/update/connect4/7x6.book')
        with open('/home/reuben/Downloads/pos_list.txt') as f:
            lines =csv.reader(f,delimiter=' ')
            self.pos_list = [line[0] for line in lines]
        self.evaluator = evaluator

    # TODO add test for evaluator without MCTS as well
    # TODO add to tensorboard?
    def test(self, num_pos=100, base_network=False, weak=False):
        games = random.sample(self.pos_list,num_pos)
        total = 0
        for game in games:
            total += self.compare(game, base_network=base_network, weak=weak)
            # print(total)
        logging.info(f"{total} correct moves out of {num_pos}: base_network = {base_network}")
        print(f"{total} correct moves out of {num_pos}: base_network = {base_network}")

    def compare(self, moves, base_network=False, weak=False):
        np_moves = np.array([int(m) for m in moves])
        reference_result = self.perfect_player.get_position_scores(np_moves, weak)
        self._text_to_board(moves)
        if not base_network:
            chosen_move = self.evaluator()
        else:
            move = self.evaluator.network(self.evaluator.root_node.state)
            chosen_move = np.argmax(move[0])
        # print(chosen_move)
        if reference_result[2][0][chosen_move] > 0:
            # If the move is judged best:
            return 1
        else:
            return 0

    def _board_to_text(self, board):
        pass

    def _text_to_board(self, text):
        self.evaluator.reset()
        player = 1
        for move in text:
            move = int(move)-1
            self.evaluator.play_action(move,player)
            player = player * -1

if __name__ == "__main__":
    print("asdf")

    network = DeepConvNetConnect4()
    network.share_memory()

    policy_gen = MCTreeSearch
    policy_args = []
    policy_kwargs = dict(iterations=400, min_memory=25000, memory_size=100000, env_gen=Connect4Env, batch_size=64,)
    policy_container = ModelContainer(policy_gen=policy_gen, policy_kwargs=policy_kwargs)

    eval = PerfectEvaluator(policy_container.setup(network=network))
    eval.test(base_network=True)
    eval.test()
    # eval.evaluator.env.step(1)
    # eval.evaluator.env.step(2,-1)



    print("asdf")