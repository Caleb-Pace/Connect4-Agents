import numpy as np
import sys

ROW_COUNT = 6
COL_COUNT = 7

class Game:
    def __init__(self):
        self.grid = np.zeros((COL_COUNT, ROW_COUNT), dtype=int)
        self.current_player = 1

        self.is_game_over = False

    def get_has_finished(self) -> bool:
        return self.is_game_over

    def print_grid(self, redraw: bool = False):
        symbols = {
            0: ".",
            1: "X",
            2: "O"
        }

        # Reposition cursor
        if redraw:
            for _ in range(ROW_COUNT + 1):
                sys.stdout.write("\033[F")  # ANSI code for cursor up
                sys.stdout.flush()

        # Print grid
        for r in range((ROW_COUNT - 1), -1, -1):
            row_str = "| "
            for c in range(COL_COUNT):
                row_str += symbols.get(self.grid[c][r]) + " "

            print(row_str + "|")
        
        # Print footer
        print("~~" + ('-'.join('-' * COL_COUNT)) + "~~")

        # Winner
        if self.is_game_over:
            print(f"Player {self.current_player} won")

    def try_drop_disc(self, col_num: int) -> bool:
        """Attempt to drop disc; Returns True if sucessful."""
        if self.is_game_over:
            return False # Game has finished

        if not self.is_valid_location(col_num):
            return False # Column is full

        self.__drop_disc(col_num)
        return True        

    def __drop_disc(self, col_num: int):
        top_row = ROW_COUNT - 1

        # Apply gravity: Find lowest free slot (in column)
        r = top_row
        while (r > 0) and (self.grid[col_num][r - 1] == 0):
            r -= 1

        # Place disc
        self.grid[col_num][r] = self.current_player
        self.__check_win()
        if not self.is_game_over:
            self.__next_player()

    def __check_win(self):
        # Horizontal check
        for r in range(ROW_COUNT):
            for c in range(COL_COUNT - 3):
                if all(self.grid[c + i][r] == self.current_player for i in range(4)):
                    self.is_game_over = True

        # Vertical check
        for c in range(COL_COUNT):
            for r in range(ROW_COUNT - 3):
                if all(self.grid[c][r + i] == self.current_player for i in range(4)):
                    self.is_game_over = True

        # Positive diagonal check (/)
        for r in range(3, ROW_COUNT):
            for c in range(COL_COUNT - 3):
                if all(self.grid[c + i][r - i] == self.current_player for i in range(4)):
                    self.is_game_over = True

        # Negative diagonal check (\)
        for r in range(ROW_COUNT - 3):
            for c in range(COL_COUNT - 3):
                if all(self.grid[c + i][r + i] == self.current_player for i in range(4)):
                    self.is_game_over = True

    def __next_player(self):
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

    def is_valid_location(self, col_num: int) -> bool:
        """Returns True if the column is not full."""
        return self.grid[col_num][ROW_COUNT - 1] == 0