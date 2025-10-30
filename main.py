from game import Game

print("Hello World!")

demo_game = Game()
demo_game.print_grid()

print(demo_game.try_drop_disc(3)) # TODO: Remove print, for debugging
demo_game.print_grid()

demo_game.try_drop_disc(3)
demo_game.print_grid(True)
