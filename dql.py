import random, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        board_size = 6 * 7

        # 2 channels for discs (Yours and Opponents).
        # 64 is arbitrary.
        # 128 is double of 64 to detect deeper patterns.
        # Padding allows NN to handle the edge of the board properly.
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),    # Search for patterns within a 3x3 window
            nn.ReLU(),                                     # Suppress deconstructive noise
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Deeper pattern search
            nn.Flatten(),                                  # Convert 2D feature map to 1D vector
            nn.Linear((128 * board_size), board_size),     # Combine feature maps into one neuron per board cell
            nn.ReLU(),                                     #
            nn.Linear(board_size, 7)                       # Map NN to Q-values (7 actions)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Credit: https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py#L25
# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class Connect4DQL():
    # TODO: Add hyperparameters as arguments
    def __init__(self):
        # Hyperparameters
        self.learning_rate = 0.001       #
        self.reward_discount_rate = 0.9  #
        self.sync_rate = 10              # How many episodes before syncing target to policy network
        self.replay_mem_size = 1000      # How many past experiences to store
        self.sample_size = 32            # How many memories to sample

        # Neural Network options
        self.loss_func = nn.MSELoss()  # TODO: Maybe look into https://en.wikipedia.org/wiki/Huber_loss
        self.optimiser = None

    def train(self):

        policy_dqn = DQN()
        target_dqn = DQN()

        pass

    def optimise(self):
        pass

    def test(self):
        pass

    # # TODO: Implement later
    # def export_model():
    #     pass
    # def import_model():
    #     pass