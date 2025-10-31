import time
from game import Game
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent

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
