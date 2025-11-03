import torch
from .agent_base import AgentBase
from game import Game
from dql import DQN, Connect4DQL

class QLearningAgent(AgentBase):
    def __init__(self, player_id: int, model_file, hidden1_size: int, hidden2_size: int):
        super().__init__(player_id)

        # Load learned policy
        self.policy_dqn = DQN(hidden1_size, hidden2_size)
        Connect4DQL.import_model(self.policy_dqn, model_file)
        self.policy_dqn.eval()  # Set model to evaluation mode

    def move(self, g: Game):
        # Best action (according to NN)
        with torch.no_grad():
            q_values = self.policy_dqn(Connect4DQL.transform_grid_to_dqn_input(g))  # Retrieve Q values

            # Select the column with the highest Q value
            for _ in range(7):
                best_action = torch.argmax(q_values).item()
                if g.try_drop_disc(best_action):
                    action = best_action
                    break  # Drop successful
                
                # Ignore full/invalid column
                q_values[0, best_action] = -float("inf")  # from batch 0
        
        return
