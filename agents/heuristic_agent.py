import random
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
        
        g.try_drop_disc(random.randint(0, 6))
        return
