from .agent_base import AgentBase
from game import Game, COL_COUNT

class HeuristicAgent(AgentBase):
    def move(self, g: Game):
        # Look for winning moves
        for c in range(COL_COUNT):
            if g.is_valid_location(c):
                # Simulate drop
                temp_grid = g.get_grid()
                g.drop_disc(temp_grid, self.player_id, c)

                # Check if win
                if g.check_win(temp_grid, self.player_id):
                    g.try_drop_disc(c) # Play winning move
                    return
                
        # Look for blocking moves (defensive)
        for c in range(COL_COUNT):
            if g.is_valid_location(c):
                # Simulate drop
                temp_grid = g.get_grid()
                g.drop_disc(temp_grid, self.opponent_id, c)

                # Check if win
                if g.check_win(temp_grid, self.opponent_id):
                    g.try_drop_disc(c) # Play blocking move
                    return
        
        # Place by preference (Prefer centre)
        column_weights = [3, 2, 4, 0, 6, 1, 5]
        for c in column_weights:
            if g.try_drop_disc(c):  # Try drop returns false on fail
                return

        raise Exception("No move made!")  # Shouldn't be reached