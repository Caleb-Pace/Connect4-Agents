from game import Game

print("\nConnect 4 with Agents:")

# Create game
demo_game = Game()
demo_game.print_grid()

## TODO: Remove, for testing
demo_game.try_drop_disc(6) # Make player 2 win

# Horizontal win
for i in range(4):
    demo_game.try_drop_disc(i)
    demo_game.try_drop_disc(i)

# # Vertical win
# for i in range(4):
#     demo_game.try_drop_disc(3)
#     demo_game.try_drop_disc(1)

# # Positive Diagonal win
# demo_game.try_drop_disc(1)
# demo_game.try_drop_disc(2)
# demo_game.try_drop_disc(2)
# demo_game.try_drop_disc(3)
# demo_game.try_drop_disc(3)
# demo_game.try_drop_disc(4)
# demo_game.try_drop_disc(3)
# demo_game.try_drop_disc(4)
# demo_game.try_drop_disc(4)
# demo_game.try_drop_disc(6)
# demo_game.try_drop_disc(4)

# # Negative Diagonal win
# demo_game.try_drop_disc(4)
# demo_game.try_drop_disc(3)
# demo_game.try_drop_disc(3)
# demo_game.try_drop_disc(2)
# demo_game.try_drop_disc(2)
# demo_game.try_drop_disc(1)
# demo_game.try_drop_disc(2)
# demo_game.try_drop_disc(1)
# demo_game.try_drop_disc(1)
# demo_game.try_drop_disc(6)
# demo_game.try_drop_disc(1)

demo_game.print_grid(True)
