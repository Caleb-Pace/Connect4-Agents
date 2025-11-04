import math, random, torch, os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F  # TODO: Remove unused
from agents.agent_base import AgentBase
from collections import deque
from datetime import datetime, timedelta  # TODO: Remove unused
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
        self.learning_rate = 0.001         # (Alpha)
        self.reward_discount_factor = 0.9  # (Gamma)
        self.sync_rate = 30                # How many actions/steps/moves before syncing target to policy network
        self.replay_mem_size = 1000        # How many past experiences to store
        self.sample_size = 32              # How many memories to sample

        # Neural Network options
        self.loss_func = nn.MSELoss()
        self.optimiser = None
        self.hidden1_size = 64       # Number of neurons in first hidden layer, (arbitrary)
        self.hidden2_size = 128      # Number of neurons in second hidden layer, (double to capture deeper patterns)
        self.checkpoint_rate = 1000  # How many episodes before creating a model checkpoint (saving the policy DQN)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)  # TODO: Remove, for debugging

        # Epsilon decay variables
        self.EPS_START = 1.0   # Maximum
        self.EPS_END   = 0.01  # Minimum
        self.EPS_DECAY = 1000  # Number of steps to decay over

        # Game variables
        self.player_id   = 0  # Initialised later
        self.opponent_id = 0  # Initialised later

        # Rewards
        self.WIN_REWARD  = 1.0
        self.LOSS_REWARD = -1.0
        self.TIE_REWARD  = 0.5

        # Data & Results
        self.training_folder = f"training/full_model/"

    def update_hyperparameters(self, gamma: float, batch_size: int, hidden1_size: int, hidden2_size: int):
        self.reward_discount_factor = gamma

        self.sample_size = batch_size

        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # Data & Results
        self.training_folder = f"training/g{self.reward_discount_factor}_b{self.sample_size}_h{self.hidden1_size}-{self.hidden2_size}_model/"

    # Adapted from: https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py#L54
    def train(self, episode_count: int, opponent_agent: AgentBase):
        # Initialise memory
        memory = ReplayMemory(self.replay_mem_size)

        # Create networks
        policy_dqn = DQN(self.hidden1_size, self.hidden2_size).to(self.device)
        target_dqn = DQN(self.hidden1_size, self.hidden2_size).to(self.device)
        target_dqn.load_state_dict(policy_dqn.state_dict())  # Copy network weights (Policy -> Target)

        # Policy network optimiser
        self.optimiser = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        # (Data) Create training folder
        if not os.path.exists(self.training_folder):
            os.makedirs(self.training_folder)

        # (Data) Create checkpoints folder
        checkpoints_folder = f"{self.training_folder}model checkpoints/"
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)

        # (Data) Compile data file - Settings
        with open(f"{self.training_folder}data.txt", "w") as file:
            file.write(f"{episode_count} episodes\n")
            
            file.write("\n< Hyperparameters >\n")
            file.write(f"Learning rate (alpha): {self.learning_rate}\n")
            file.write(f"Learning rate (Gamma): {self.reward_discount_factor}\n")
            file.write(f"            Sync rate: {self.sync_rate}\n")
            file.write(f"   Replay memory size: {self.replay_mem_size}\n")
            file.write(f"           Batch size: {self.sample_size}\n")

            file.write("\n< Neural Network >\n")
            file.write(f"Hidden layer 1 neurons: {self.hidden1_size}\n")
            file.write(f"Hidden layer 2 neurons: {self.hidden2_size}\n")
            file.write(f"         Loss function: {self.loss_func.__class__.__name__}\n")
            file.write(f"             Optimiser: {self.optimiser.__class__.__name__}\n")
            file.write(f"                Device: {self.device}\n")

            file.write("\n< Reward weights >\n")
            file.write(f" Win: {self.WIN_REWARD}\n")
            file.write(f"Loss: {self.LOSS_REWARD}\n")
            file.write(f" Tie: {self.TIE_REWARD}\n")

        # Track episode rewards
        episode_wins    = np.zeros(episode_count)  # For optimisation check
        episode_rewards = np.zeros(episode_count)  # (Data) For win rate
        
        # (Data) Track actions to win
        actions_to_win = []

        # (Data) Track TD/DQN loss
        loss_values = []

        # Initialise epsilon for epsilon-greedy exploration
        epsilon = self.EPS_START                   # Random action percentage
        epsilon_history = np.zeros(episode_count)  # (Data) For plotting epsilon

        # Track actions/steps/moves for network syncing
        unsynced_actions = 0

        # (Data) Track training time
        start_time = datetime.now()

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
                memory.push((grid, connect4.get_grid(), self.player_id, action, reward, connect4.get_has_finished()))

                # Opponent move (if second)
                if self.opponent_id == 2:
                    opponent_agent.move(connect4)

            # Record wins
            if reward == self.WIN_REWARD:
                episode_wins[i] = 1

                # (Data) Calculate action count
                total_moves = connect4.get_move_count()
                if self.player_id == 1:
                    total_moves += 1  # Player 2 hasn't moved yet - Complete the round
                actions_to_win.append((total_moves // 2))

            # Record reward
            episode_rewards[i] = reward

            # (Data) Record epsilon value
            epsilon_history[i] = epsilon

            # Epsilon decay (Action choice strategy)
            epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-i / self.EPS_DECAY)

            # Only improve network if there is enough experience and at least 1 win
            if (len(memory) > self.sample_size) and (np.sum(episode_wins) > 0):
                # Optimise network
                sample = memory.sample(self.sample_size)
                loss_values.append(self.optimise(sample, policy_dqn, target_dqn))

                # Sync networks
                if unsynced_actions >= self.sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())  # Copy network weights (Policy -> Target)
                    unsynced_actions = 0

            # Model checkpoint
            if (i % self.checkpoint_rate == 0) and (i > 0) and (i < episode_count):
                self.export_model(policy_dqn, f"{checkpoints_folder}e{i}.pt")

        # (Data) Track training time
        end_time = datetime.now()

        # Export policy (network)
        self.export_model(policy_dqn, f"{self.training_folder}model.pt")

        # (Data) Calculate training time
        duration = end_time - start_time

        # (Data) Compile data file - Results and training time
        recent_window = min(100, len(episode_rewards))  # Handle <100 episodes
        with open(f"{self.training_folder}data.txt", "a") as file:
            file.write("\n< Statistics >\n")
            recent_losses = loss_values[-recent_window:]
            average_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            file.write(f" DQN loss: {average_loss}\n")
            file.write(f" Win rate: {np.sum(episode_rewards[-recent_window:] == self.WIN_REWARD) / recent_window}\n")
            file.write(f"Loss rate: {np.sum(episode_rewards[-recent_window:] == self.LOSS_REWARD) / recent_window}\n")
            file.write(f" Tie rate: {np.sum(episode_rewards[-recent_window:] == self.TIE_REWARD) / recent_window}\n")

            file.write(f"Training time elapsed: {format_duration(duration)}\n")

        # (Results) Plot rolling Win rate
        bin_size = 100
        num_bins = episode_count // bin_size
        binned_win_rate = [
            np.mean(episode_wins[i*bin_size:(i+1)*bin_size]) * 100
            for i in range(num_bins)
        ]
        bin_labels = [(i+1)*bin_size for i in range(num_bins)]
        plt.figure(figsize=(10,4))
        plt.plot(bin_labels, binned_win_rate, marker='o', color="green", label="Win Rate (%)")
        plt.title(f"Win Rate per {bin_size} Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.training_folder}win_rate.png", dpi=300)
        plt.close()

        # (Results) Plot TD/DQN loss
        plt.figure(figsize=(10,4))
        plt.plot(loss_values, color="red", label="TD Loss")
        plt.title("TD Loss over Optimization Steps")
        plt.xlabel("Step")
        plt.ylabel("TD Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.training_folder}td_loss.png", dpi=300)
        plt.close()

        # (Results) Plot Moves to win
        window = 10  # Episode window
        if len(actions_to_win) >= window:
            actions_ma = np.convolve(actions_to_win, np.ones(window)/window, mode="valid")

            plt.figure(figsize=(10,4))
            plt.plot(actions_to_win, alpha=0.5, label="Moves per Win")
            plt.plot(range(window-1, len(actions_to_win)), actions_ma, color="blue", label=f"{window}-Episode MA")
            plt.title("Moves per Win with Moving Average")
            plt.xlabel("Episode")
            plt.ylabel("Moves")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.training_folder}moves_per_win.png", dpi=300)
            plt.close()

        # (Results) Plot Epsilon decay
        plt.figure(figsize=(10,4))
        plt.plot(epsilon_history, color="purple", label="Epsilon")
        plt.title("Epsilon Decay over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.training_folder}epsilon_decay.png", dpi=300)
        plt.close()

        # (Data) Define game states
        demos = [
            (1, np.zeros((7, 6), dtype=int)),  # Empty grid (Shows column preference)
            (1, np.array([
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])),  # P1 - Offense or Defense
            (2, np.array([
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])),  # P2 - Offense or Defense
            (2, np.array([
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 2, 1, 0, 0, 0],
                [2, 1, 1, 0, 0, 0],
                [1, 2, 1, 0, 0, 0],
                [2, 2, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0]
            ]))  # Double threat
        ]

        # (Results) Plot Q-values
        action_labels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        for i, (player_id, grid) in enumerate(demos):
            with torch.no_grad():
                q_values = policy_dqn(self.transform_grid_to_dqn_input(grid, player_id, (2 if player_id == 1 else 1)))  # Retrieve Q values

                # Move tensor to CPU and convert to numpy
                q_values_np = q_values.cpu().numpy()

            plt.figure(figsize=(8,2))
            sns.heatmap(
                q_values_np,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                xticklabels=action_labels,
                yticklabels=False,
                cbar=True
            )
            plt.title(f"Q-values for State {i}")
            plt.xlabel("Actions")
            plt.ylabel("State")
            plt.tight_layout()
            plt.savefig(f"{self.training_folder}q_values_state_{i}.png", dpi=300)
            plt.close()

    def transform_grid_to_dqn_input(self, grid: np.ndarray, own_id: int, opp_id: int):
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
    def optimise(self, sample, policy_dqn: DQN, target_dqn: DQN) -> float:
        current_q_list = []
        target_q_list = []

        for grid, new_grid, player_id, action, reward, has_finished in sample:
            
            # Get opponent ID for dqn input
            opponent_id = 2 if player_id == 1 else 1

            # Get target Q value
            if has_finished:
                # Terminal state: Q-target equals the immediate reward (no future discounted Q)
                target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            else:
                # Calculate target Q value 
                with torch.no_grad():
                    next_q = target_dqn(
                        self.transform_grid_to_dqn_input(new_grid, player_id, opponent_id)
                    ).max()

                    target = torch.tensor(
                        [reward + (self.reward_discount_factor * next_q)],
                        dtype=torch.float32,
                        device=self.device
                    )

            dqn_input = self.transform_grid_to_dqn_input(grid, player_id, opponent_id)

            # Get the Q value sets
            current_q = policy_dqn(dqn_input)
            current_q_list.append(current_q)  # For calculating loss
            target_q = target_dqn(dqn_input)

            # Update target Q for the taken action (with the computed Bellman target)
            target_q[0, action] = target  # in batch 0
            target_q_list.append(target_q)

        # Compute loss for the sample
        loss = self.loss_func(torch.stack(current_q_list), torch.stack(target_q_list))
        loss_value = loss.item()  # Convert tensor to a float

        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss_value  # (Data) For plotting loss

    # Adapted from: https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py#L208
    def test(self, model_file, hidden1_size: int, hidden2_size: int, episode_count: int, opponent_agent: AgentBase):
        # Load learned policy
        policy_dqn = DQN(hidden1_size, hidden2_size).to(self.device)
        self.import_model(policy_dqn, model_file)
        policy_dqn.eval()  # Set model to evaluation mode
        
        # Data & Results
        model_name = os.path.splitext(model_file)
        self.testing_folder  = f"testing/dql_vs_{opponent_agent.__class__.__name__}/{model_name}/"
        match_example_frequency = episode_count // 5

        # (Data) Track episode rewards
        episode_rewards = np.zeros(episode_count)  # (Data) For game results
        episode_wins    = np.zeros(episode_count)  # (Data) For plotting wins
        
        # (Data) Track actions to win
        actions_to_win = []

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

                # Check reward
                reward = self.calculate_reward(connect4)

                # Opponent move (if second)
                if self.opponent_id == 2:
                    opponent_agent.move(connect4)

            # (Data) Record win
            if reward == self.WIN_REWARD:
                episode_wins[i] = 1

                # Calculate action count
                total_moves = connect4.get_move_count()
                if self.player_id == 1:
                    total_moves += 1  # Player 2 hasn't moved yet - Complete the round
                actions_to_win.append((total_moves // 2))
            
            # (Data) Record reward
            episode_rewards[i] = reward

            # (Data) Save match examples
            if (i % match_example_frequency == 0) and (i > 0):
                with open(f"{self.testing_folder}", "a") as file:
                    file.write(f"{connect4.grid_to_string()}\n")

        # (Results) Plot game results
        wins = np.sum(episode_rewards == self.WIN_REWARD)
        ties = np.sum(episode_rewards == self.TIE_REWARD)
        losses = np.sum(episode_rewards == self.LOSS_REWARD)
        plt.figure(figsize=(6,4))
        plt.bar(["Wins", "Ties", "Losses"], [wins, ties, losses], color=["green", "gray", "red"])
        plt.title("Episode Results Totals")
        plt.ylabel("Number of Episodes")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("episode_results_totals.png", dpi=300)
        plt.show()

        # (Results) Plot rolling Win rate
        bin_size = 100
        num_bins = episode_count // bin_size
        binned_win_rate = [
            np.mean(episode_wins[i*bin_size:(i+1)*bin_size]) * 100
            for i in range(num_bins)
        ]
        bin_labels = [(i+1)*bin_size for i in range(num_bins)]
        plt.figure(figsize=(10,4))
        plt.plot(bin_labels, binned_win_rate, marker='o', color="green", label="Win Rate (%)")
        plt.title(f"Win Rate per {bin_size} Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.testing_folder}win_rate.png", dpi=300)
        plt.close()

        # (Results) Plot Moves to win
        window = 10  # Episode window
        if len(actions_to_win) >= window:
            actions_ma = np.convolve(actions_to_win, np.ones(window)/window, mode="valid")
        else:
            actions_ma = actions_to_win  # Not enough wins for rolling average
        plt.figure(figsize=(10,4))
        plt.plot(actions_to_win, alpha=0.5, label="Moves per Win")
        plt.plot(range(window-1, len(actions_to_win)), actions_ma, color="blue", label=f"{window}-Episode MA")
        plt.title("Moves per Win with Moving Average")
        plt.xlabel("Episode")
        plt.ylabel("Moves")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.testing_folder}moves_per_win.png", dpi=300)
        plt.close()

    def export_model(self, policy_dqn: DQN, model_file):
        torch.save(policy_dqn.state_dict(), model_file)

    def import_model(self, policy_dqn: DQN, model_file):
        policy_dqn.load_state_dict(torch.load(model_file))


def format_duration(duration):
    total_seconds = duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"