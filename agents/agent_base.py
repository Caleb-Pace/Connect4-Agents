from abc import ABC, abstractmethod
from game import Game

class AgentBase(ABC):
    def __init__(self, player_id: int):
        self.is_player1 = player_id == 1
        
        self.player_id = player_id
        self.opponent_id = 2 if self.is_player1 else 1
    
    @abstractmethod
    def move(self, g: Game):
        """
        Execute the agent's move on the given game.
        
        Args:
            g (Game): The current Connect 4 game instance that provides
                hooks for move validation, disc placement, and board state access.
        """
        pass
