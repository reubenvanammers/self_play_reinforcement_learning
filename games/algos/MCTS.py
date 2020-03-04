import numpy as np
from anytree import NodeMixin
from rl_utils.memory import Memory
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "done", "next_state")) #TODO - make transition relevant to MCTS



class MCNode(NodeMixin):

    def __init__(self, state=None, n=0, w=0, p=0, parent=None):
        self.state = state
        self.n = n
        self.w = w
        self.p = p
        self.parent = parent

        self.cpuct = 1

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

    # def prune(self, action = None):
    #     # Only keep the specified action: delete the ot
    #     if action:

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

        self.temp_memory = []

        self.temperature_cutoff = temperature_cutoff
        self.actions = actions
        self.moves_played = 0

    def search_and_play(self):
        temperature = 1 if self.moves_played < self.temperature_cutoff else 0
        self.search()
        move = self.play(temperature)
        return move

    def update(self, s, a, r, done, next_s):
        #TODO - have temperorary queue and push to real queue when games are complete
        if not done:
            self.

    def play(self, temp=0.01):
        play_probs = [child.play_prob(temp) for child in self.root_node.children]
        choice = np.random.choice(self.actions, p=play_probs)
        self.root_node.children = self.root_node.children[choice]
        self.root_node = self.root_node.children[0]

        self.moves_played += 1
        return choice

    def expand_node(self, node, choice, player=1):
        parent_state = node.parent.state
        env = self.env_gen()
        env.set_state(parent_state)
        s, r, done, _ = env.step(choice, player=player)
        probs, v = self.evaluator(s)
        node.create_children(probs)
        return v


def search(self):
    node = self.root_node
    for i in range(self.iterations):
        while not node.is_leaf:
            select_probs = [child.select_prob for child in node.children]
            choice = np.random.choice(self.actions, p=select_probs)
            node = node.children[choice]
        v = self.expand_node(node, choice)
        node.backup(v)
