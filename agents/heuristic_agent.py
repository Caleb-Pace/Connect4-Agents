from .agent_base import AgentBase
from game import Game

class HeuristicAgent(AgentBase):
    def __init__(self, player_id: int):
        self.is_player1 = player_id == 1

    def move(self, g: Game):
        # g.try_drop_disc(random.randint(0, 6)) # TODO: Remove, for debugging

        if self.is_player1:
            g.try_drop_disc(3)
        else:
            g.try_drop_disc(2)

        return
    
