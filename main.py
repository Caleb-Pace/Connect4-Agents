import time
import numpy as np
from game import Game
from agents.heuristic_agent import HeuristicAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from datetime import datetime
from dql import Connect4DQL

def main():
    # export_demos()
    # play()

    train_models()

    # test_model(f"training/full_model/model checkpoints/" + "e2000" + ".pt", 64, 128)

    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]}] Jobs complete!")

def train_models():
    gammas = [0.9, 0.99]           # Reward discount factor (Short-term vs Long-term strategy) 
    batch_sizes = [32, 64]         # Memory sample size
    hidden1_sizes = [64, 32, 64]   # Number of neurons in first hidden layer, (arbitrary)
    hidden2_sizes = [128, 64, 64]  # Number of neurons in second hidden layer, (to capture deeper patterns)

    opp_agent = HeuristicAgent(1)  # ID will be changed

    connect4_dql = Connect4DQL()

    # Full convergence
    episodes = 100_000
    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]}] Training model to full convergence (over {episodes} episodes)")
    connect4_dql.train(episodes, opp_agent)

    # Hyperparameter sweep
    episodes = 10_000
    parameter_set = 0
    for gamma in gammas:
        for batch_size in batch_sizes:
            for i in range(len(hidden1_sizes)):
                print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]}] Training model with hyperparameter set {parameter_set} (over {episodes} episodes)")
                parameter_set += 1

                connect4_dql.update_hyperparameters(gamma, batch_size, hidden1_sizes[i], hidden2_sizes[i])

                connect4_dql.train(episodes, opp_agent)

def test_model(model_file, hidden1_size: int, hidden2_size: int):
    opp_agent = HeuristicAgent(1)  # ID will be changed
    connect4_dql = Connect4DQL()
    episodes = 1_000

    # Test pre-trained model
    print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]}] Testing model \"{model_file}\" (for {episodes} episodes)")
    connect4_dql.test(model_file, hidden1_size, hidden2_size, episodes, opp_agent)

def export_demos():
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

    for i, (player_id, grid) in enumerate(demos):
        g = Game()
        g.set_grid(grid)

        with open("demos.txt", "a") as file:
            file.write(f"   < State {i} >\n")
            file.write(g.grid_to_string(False)[:-1])
            file.write(f"Agent is player {player_id}\n\n")

def play():
    print("\nConnect 4 with Agents:")

    # Create game
    g = Game()
    g.print_grid()

    # Create agents
    agent1 = HeuristicAgent(1)
    agent2 = HeuristicAgent(2)

    # Run game
    while not g.get_has_finished():
        agent1.move(g)
        agent2.move(g)

        g.print_grid(True)
        time.sleep(0.1)


if __name__ == "__main__":
    main()