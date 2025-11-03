import random, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from agents.agent_base import AgentBase
from collections import deque
from game import Game


class DQN(nn.Module):
    def __init__(self, h1_size: int, h2_size: int):
        super().__init__()

        board_size = 6 * 7

        # 2 channels for discs (Yours and Opponents).
        # Padding allows NN to handle the edge of the board properly.
        self.model = nn.Sequential(
            nn.Conv2d(2, h1_size, kernel_size=3, padding=1),        # Search for patterns within a 3x3 window
            nn.ReLU(),                                              # Suppress deconstructive noise
            nn.Conv2d(h1_size, h2_size, kernel_size=3, padding=1),  # Deeper pattern search
            nn.Flatten(),                                           # Convert 2D feature map to 1D vector
            nn.Linear((h2_size * board_size), board_size),          # Combine feature maps into one neuron per board cell
            nn.ReLU(),                                              #
            nn.Linear(board_size, 7)                                # Map NN to Q-values (7 actions)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Credit: https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py#L25
# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def push(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class Connect4DQL():
    def __init__(self):
        # Hyperparameters
        self.learning_rate = 0.001       # (Alpha)
        self.reward_discount_rate = 0.9  # (Gamma)
        self.sync_rate = 10              # How many actions/steps/moves before syncing target to policy network
        self.replay_mem_size = 1000      # How many past experiences to store
        self.sample_size = 32            # How many memories to sample

        # Neural Network options
        self.loss_func = nn.MSELoss()
        self.optimiser = None
        self.device = "cpu"
        self.hidden1_size = 64   # Number of neurons in first hidden layer, (arbitrary)
        self.hidden2_size = 128  # Number of neurons in second hidden layer, (double to capture deeper patterns)
        
        # Game variables
        self.player_id   = 0  # Initialised later
        self.opponent_id = 0  # Initialised later

        # Rewards
        self.WIN_REWARD  = 1.0
        self.LOSS_REWARD = -1.0
        self.TIE_REWARD  = 0.5

    def update_hyperparameters(self, gamma: float, batch_size: int, hidden1_size: int, hidden2_size: int):
        self.reward_discount_rate = gamma

        self.sample_size = batch_size

        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

    # Adapted from: https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py#L54
    def train(self, episode_count: int, opponent_agent: AgentBase):
        # Initialise memory
        memory = ReplayMemory(self.replay_memory_size)
        
        # Initialise epsilon for epsilon-greedy exploration
        epsilon = 1.0 # Random action percentage
        
        # Create networks
        policy_dqn = DQN(self.hidden1_size, self.hidden2_size)
        target_dqn = DQN(self.hidden1_size, self.hidden2_size)
        target_dqn.load_state_dict(policy_dqn.state_dict())  # Copy network weights (Policy -> Target)

        # Policy network optimiser
        self.optimiser = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        # Track episode rewards
        episode_rewards = np.zeros(episode_count)

        # Track actions/steps/moves for network syncing
        unsynced_actions = 0

        # Create game
        connect4 = Game()

        for i in range(episode_count):
            connect4.reset()  # Reset game
            reward = 0

            # Switch who goes first
            if i % 2 == 0:
                # Play as player 1
                self.player_id   = 1
                self.opponent_id = 2
            else:
                # Play as player 2
                self.player_id   = 2
                self.opponent_id = 1
            opponent_agent.update_player_id(self.opponent_id)  # Update player ID for opponent agent
            
            # Agent learns to plays Connect 4
            while not connect4.get_has_finished():
                
                # Opponent move (if first)
                if self.opponent_id == 1:
                    opponent_agent.move(connect4)

                # Select action using the epsilon-greedy strategy
                grid = connect4.get_grid()
                action = None  # Initialised later
                if random.random() < epsilon:
                    # Random action
                    columns = [0, 1, 2, 3, 4, 5, 6]
                    
                    # Select column randomly
                    while len(columns) > 0:
                        rnd_action = random.choice(columns)
                        if connect4.try_drop_disc(rnd_action):
                            action = rnd_action
                            break  # Drop successful
                        
                        # Remove full/invalid column
                        columns.remove(rnd_action)
                else:
                    # Best action (according to NN)
                    with torch.no_grad():
                        q_values = policy_dqn(self.transform_grid_to_dqn_input(grid, self.player_id, self.opponent_id))  # Retrieve Q values

                        # Select the column with the highest Q value
                        for _ in range(7):
                            best_action = torch.argmax(q_values).item()
                            if connect4.try_drop_disc(best_action):
                                action = best_action
                                break  # Drop successful
                            
                            # Ignore full/invalid column
                            q_values[0, best_action] = -float("inf")  # from batch 0

                # Increment action unsynced counter
                unsynced_actions += 1
                
                # Check reward
                reward = self.calculate_reward(connect4)

                # Remember experience
                memory.append(grid, connect4.get_grid(), self.player_id, action, reward, connect4.get_has_finished())

                # Opponent move (if second)
                if self.opponent_id == 2:
                    opponent_agent.move(connect4)

            # Record wins
            if reward == self.WIN_REWARD:
                episode_rewards[i] = 1

            # Only improve network if there is enough experience and at least 1 win
            if (len(memory) > self.sample_size) and (np.sum(episode_rewards) > 0):
                # Optimise network
                sample = memory.sample(self.sample_size)
                self.optimise(sample, policy_dqn, target_dqn)
                
                # Epsilon decay (Action choice strategy)
                epsilon = max(((epsilon - 1) / episode_count), 0)

                # Sync networks
                if unsynced_actions >= self.sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())  # Copy network weights (Policy -> Target)
                    unsynced_actions = 0

        # Export policy (network)
        self.export_model(policy_dqn)

    def transform_grid_to_dqn_input(self, game: Game, own_id: int, opp_id: int):
        grid = game.get_grid()
        own_discs = (grid == own_id).astype(np.float32)
        opp_discs = (grid == opp_id).astype(np.float32)
        
        stacked = np.stack([own_discs, opp_discs])
        return torch.tensor(stacked, dtype=torch.float32, device=self.device).unsqueeze(0)  # unsqueeze: Add batch_size dimension (1 grid per batch)
    
    def calculate_reward(self, game: Game) -> float:
        if not game.get_has_finished():
            return 0.0

        # Determine the outcome
        grid = game.get_grid()

        if game.check_win(grid, self.player_id):
            return self.WIN_REWARD
        if game.check_win(grid, self.opponent_id):
            return self.LOSS_REWARD
        
        return self.TIE_REWARD

    # Adapted from: https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py#L156
    def optimise(self, sample, policy_dqn: DQN, target_dqn: DQN):
        current_q_list = []
        target_q_list = []

        for grid, new_grid, player_id, action, reward, has_finished in sample:
            
            # Get opponent ID for dqn input
            opponent_id = 2 if player_id == 1 else 1

            # Get target Q value
            if has_finished:
                # Terminal state: Q-target equals the immediate reward (no future discounted Q)
                target = torch.FloatTensor([reward])
            else:
                # Calculate target Q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + (self.reward_discount_rate * target_dqn(self.transform_grid_to_dqn_input(new_grid, player_id, opponent_id)).max())
                    )

            dqn_input = self.transform_grid_to_dqn_input(grid, player_id, opponent_id)

            # Get the Q value sets
            current_q = policy_dqn(dqn_input)
            current_q_list.append(current_q)  # For calculating loss
            target_q = target_dqn(dqn_input)

            # Update target Q for the taken action (with the computed Bellman target)
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the sample
        loss = self.loss_func(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    # Adapted from: https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py#L208
    def test(self, model_file, hidden1_size: int, hidden2_size: int, episode_count: int, opponent_agent: AgentBase):
        # Load learned policy
        policy_dqn = DQN((self.hidden1_size, self.hidden2_size))
        self.import_model(policy_dqn, model_file)
        policy_dqn.eval()  # Set model to evaluation mode
        
        # Create game
        connect4 = Game()    

        # Test model
        for i in range(episode_count):
            connect4.reset()  # Reset game

            # Switch who goes first
            if i % 2 == 0:
                # Play as player 1
                self.player_id   = 1
                self.opponent_id = 2
            else:
                # Play as player 2
                self.player_id   = 2
                self.opponent_id = 1
            opponent_agent.update_player_id(self.opponent_id)  # Update player ID for opponent agent

            # Agent plays Connect 4
            while not connect4.get_has_finished():
                
                # Opponent move (if first)
                if self.opponent_id == 1:
                    opponent_agent.move(connect4)

                # Select best action (according to NN)
                grid = connect4.get_grid()
                with torch.no_grad():
                    q_values = policy_dqn(self.transform_grid_to_dqn_input(grid, self.player_id, self.opponent_id))  # Retrieve Q values

                    # Select the column with the highest Q value
                    for _ in range(7):
                        best_action = torch.argmax(q_values).item()
                        if connect4.try_drop_disc(best_action):
                            action = best_action
                            break  # Drop successful
                        
                        # Ignore full/invalid column
                        q_values[0, best_action] = -float("inf")  # from batch 0

                # Opponent move (if second)
                if self.opponent_id == 2:
                    opponent_agent.move(connect4)

    def export_model(self, policy_dqn: DQN):
        torch.save(policy_dqn.state_dict(), "")

    def import_model(self, policy_dqn: DQN, model_file):
        policy_dqn.load_state_dict(torch.load(model_file))