import numpy as np
import sys

ROW_COUNT = 6
COL_COUNT = 7

class Game:
    def __init__(self):
        self.grid = np.zeros((COL_COUNT, ROW_COUNT), dtype=int)  # Column-major
        self.current_player = 1

        self.total_move_count = 0
        self.is_game_over = False

    def get_grid(self) -> np.ndarray:
        """Returns a copy of the grid"""
        return self.grid.copy()

    def get_has_finished(self) -> bool:
        return self.is_game_over

    def get_move_count(self) -> int:
        return self.total_move_count

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
            print(f"Player {self.current_player} won (on move {self.total_move_count})")

    def try_drop_disc(self, col_num: int) -> bool:
        """Attempt to drop disc; Returns True if sucessful."""
        if self.is_game_over:
            return False # Game has finished

        if not (0 <= col_num < COL_COUNT):
            raise Exception(f"Column {col_num} is out of range [{0}, {COL_COUNT})")
            # return False # Invalid column number

        if not self.is_valid_location(col_num):
            return False # Column is full

        # Place disc
        self.drop_disc(self.grid, self.current_player, col_num)
        self.total_move_count += 1

        # Check if game has ended
        if self.total_move_count >= (ROW_COUNT * COL_COUNT) or self.check_win(self.grid, self.current_player):
            self.is_game_over = True  # Tie or Win
        else:
            self.__next_player()      # Continue - Next players turn

        return True        

    def drop_disc(self, grid: np.ndarray, player_id: int, col_num: int):
        top_row = ROW_COUNT - 1

        # Apply gravity: Find lowest free slot (in column)
        r = top_row
        while (r > 0) and (self.grid[col_num][r - 1] == 0):
            r -= 1

        # Place disc
        grid[col_num][r] = player_id

    def check_win(self, grid: np.ndarray, player_id: int) -> bool:
        cols, rows = grid.shape  # Column-major

        # Horizontal check
        for c in range(cols - 3):
            for r in range(rows):
                if np.all(grid[c:c+4, r] == player_id):
                    return True
        
        # Vertical check
        for c in range(cols):
            for r in range(rows - 3):
                if np.all(grid[c, r:r+4] == player_id):
                    return True
        
        # Positive diagonal check (/)
        for c in range(cols - 3):
            for r in range(rows - 3):
                if all(grid[c + i, r + i] == player_id for i in range(4)):
                    return True
        
        # Negative diagonal check (\)
        for c in range(cols - 3):
            for r in range(3, rows):
                if all(grid[c + i, r - i] == player_id for i in range(4)):
                    return True
        
        return False

    def __next_player(self):
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

    def is_valid_location(self, col_num: int) -> bool:
        """Returns False if the column is full."""
        return self.grid[col_num][ROW_COUNT - 1] == 0