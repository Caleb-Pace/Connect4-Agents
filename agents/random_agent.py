import random
from .agent_base import AgentBase
from game import Game

class RandomAgent(AgentBase):
    def move(self, g: Game):
        g.try_drop_disc(random.randint(0, 6))
        return
