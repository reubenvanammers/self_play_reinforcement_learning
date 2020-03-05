import numpy as np
from anytree import NodeMixin
from rl_utils.memory import Memory
from collections import namedtuple

Move = namedtuple("Move", ("state", "action", "predicted_val", "actual_val", "network_probs", "tree_probs"))


class MCNode(NodeMixin):

    def __init__(self, state=None, n=0, w=0, p=0, parent=None):
        self.state = state
        self.n = n
        self.w = w
        self.p = p
        self.parent = parent

        self.cpuct = 1

        self.v = None

    @property
    def q(self):
        return self.w / self.n if self.n else 0

    @property
    def u(self):
        return self.cpuct * self.p * np.sqrt(self.parent.n) / (1 + self.n)

    @property
    def select_prob(self):
        return self.q + self.u

    def backup(self, v):
        self.w += v
        self.n += 1
        if self.parent:
            self.parent.backup(v)

    def play_prob(self, temp):
        return np.power(self.n, 1 / temp)

    def create_children(self, action_probs):
        children_list = []
        for action_prob in action_probs:
            children_list.append(MCNode(p=action_prob), parent=self)
        self.children = children_list

    def _post_detach_children(self, children):
        for child in children:
            if child.children:
                child.children = []
            del child


class MCTreeSearch:

    def __init__(self, iterations, evaluator, env_gen, actions=7, temperature_cutoff=5):
        self.iterations = iterations
        self.evaluator = evaluator
        self.env_gen = env_gen
        self.env = env_gen()
        base_state = self.env.reset()
        self.root_node = MCNode(state=base_state)

        # self.initial_probs, self.v = self.evaluator(base_state)

        self.temp_memory = []
        self.memory = Memory()

        self.temperature_cutoff = temperature_cutoff
        self.actions = actions
        self.moves_played = 0

    def search_and_play(self):
        temperature = 1 if self.moves_played < self.temperature_cutoff else 0
        # self.initial_probs, self.v = self.evaluator(base_state)
        self.search()
        move = self.play(temperature)
        return move

    def update(self, s, a, r, done, next_s):
        if done:
            for experience in self.temp_memory:
                experience.actual_val = r
                self.memory.add(experience)
            self.temp_memory = []
        # TODO atm start of with running update, then maybe move to async model like in paper

    def play(self, temp=0.01):
        play_probs = [child.play_prob(temp) for child in self.root_node.children]
        move_probs = [child.p for child in self.root_node.children]

        choice = np.random.choice(self.actions, p=play_probs)
        self.root_node.children = self.root_node.children[choice]
        self.root_node = self.root_node.children[0]

        self.moves_played += 1

        self.temp_memory.append(Move(self.root_node.state, choice, self.root_node.v, None, move_probs, play_probs))
        return choice

    def expand_node(self, parent_node, choice, player=1):
        env = self.env_gen()
        env.set_state(parent_node.state)
        s, r, done, _ = env.step(choice, player=player)
        if done:
            v = r
            child_node = parent_node.children[choice]
        else:
            # TODO add if done set MCTS to appropriate val
            probs, v = self.evaluator(s)
            child_node = parent_node.children[choice]
            child_node.create_children(probs)
        return child_node, v

    def search(self):
        node = self.root_node
        for i in range(self.iterations):
            while True:
                select_probs = [child.select_prob for child in node.children]
                choice = np.random.choice(self.actions, p=select_probs)
                if node.is_leaf:
                    node, v = self.expand_node(node, choice)
                    node.backup(v)
                    node.v = v
                    break
                else:
                    node = node.children[choice]
