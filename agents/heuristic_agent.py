import random
from .agent_base import AgentBase
from game import Game

class HeuristicAgent(AgentBase):
    def move(self, g: Game):
        g.try_drop_disc(random.randint(0, 6)) # TODO: Remove, for debugging
        return
