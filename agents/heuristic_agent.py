import numpy as np
from .agent_base import AgentBase
import random

class HeuristicAgent(AgentBase):
    def move(self, current_grid: np.ndarray) -> int:
        return random.randint(0, 6) # TODO: Remove, for debugging
