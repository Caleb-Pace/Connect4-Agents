import time, random
from game import Game

print("\nConnect 4 with Agents:")

# Create game
demo_game = Game()
demo_game.print_grid()

# Run game
while not demo_game.get_has_finished():
    demo_game.try_drop_disc(random.randint(0, 6))

    demo_game.print_grid(True)
    time.sleep(0.1)
