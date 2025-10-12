"""
Heuristic control functions for Breakout.

Provides AI agent that uses ball-following strategy to play the game.
"""

"""
    get_heuristic_action(game_state) -> Int

Get paddle movement action using ball-following heuristic strategy.

Implements a simple but effective strategy: move the paddle toward the ball's
horizontal position with a small dead zone to reduce jittery movement.

# Arguments
- `game_state`: Current game state tuple

# Returns
- `-1`: Move paddle left (ball is to the left)
- `1`: Move paddle right (ball is to the right)
- `0`: No movement (ball is approximately centered)
"""
function get_heuristic_action(game_state)
    # Extract game state components
    score, ball_cx, ball_cy, ball_vx, ball_vy, paddle_cx, bricks = game_state
    
    # Ball-following heuristic with small dead zone
    if ball_cx < paddle_cx - 1
        return -1  # Move left
    elseif ball_cx > paddle_cx + 1
        return 1   # Move right
    else
        return 0   # Stay in place (dead zone)
    end
end

