import numpy as np
from abc import ABC, abstractmethod

class AgentBase(ABC):
    @abstractmethod
    def move(self, current_grid: np.ndarray) -> int:
        """
        Given the current board state, return the column index (0-based)
        where the agent wants to place its piece.
        
        :param current_grid: 2D list representing the Connect 4 board
        :return: int (column index)
        """
        pass