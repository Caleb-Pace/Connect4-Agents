import time
from game import Game
from agents.heuristic_agent import HeuristicAgent

print("\nConnect 4 with Agents:")

# Create game
g = Game()
g.print_grid()

# Create agents
agent1 = HeuristicAgent()
agent2 = HeuristicAgent()

# Run game
while not g.get_has_finished():
    chosen_col = agent1.move(g.get_grid())
    if not g.try_drop_disc(chosen_col):
        if not g.get_has_finished():
            raise Exception("Invalid disc drop")

    chosen_col = agent2.move(g.get_grid())
    if not g.try_drop_disc(chosen_col):
        if not g.get_has_finished():
            raise Exception("Invalid disc drop")

    g.print_grid(True)
    time.sleep(0.1)
