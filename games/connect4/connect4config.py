from games.algos.mcts import MCTreeSearch
from games.general.base_model import ModelContainer

policy_kwargs = dict(iterations=800, min_memory=25000, memory_size=300000, batch_size=128,)

mcts = ModelContainer(MCTreeSearch, policy_kwargs=policy_kwargs)
