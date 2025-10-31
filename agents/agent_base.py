from abc import ABC, abstractmethod
from game import Game

class AgentBase(ABC):
    @abstractmethod
    def move(self, g: Game):
        """
        Execute the agent's move on the given game.
        
        Args:
            g (Game): The current Connect 4 game instance that provides
                hooks for move validation, disc placement, and board state access.
        """
        pass