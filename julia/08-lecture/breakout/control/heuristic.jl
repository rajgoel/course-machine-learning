"""
Heuristic control module for Breakout.

Provides AI agent that uses ball-following strategy to play the game.
"""
module Control

using GameZero
using ..GameLogic

export get_action

"""
    get_action(g) -> Int

Get paddle movement action using ball-following heuristic strategy.

Implements a simple but effective strategy: move the paddle toward the ball's
horizontal position with a small dead zone to reduce jittery movement.

# Returns
- `-1`: Move paddle left (ball is to the left)
- `1`: Move paddle right (ball is to the right)
- `0`: No movement (ball is approximately centered)
"""
function get_action(g)
    # Get current game state
    game_state = get_game_state()
    score, ball_x, ball_y, ball_vx, ball_vy, paddle_x, bricks = game_state
    
    # Ball-following heuristic with small dead zone
    if ball_x < paddle_x - 1
        return -1  # Move left
    elseif ball_x > paddle_x + 1
        return 1   # Move right
    else
        return 0   # Stay in place (dead zone)
    end
end

end # module Control
