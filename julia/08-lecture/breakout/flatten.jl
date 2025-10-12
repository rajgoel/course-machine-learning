"""
Functions for flattening the game state.

Converts game state into flattened vectors suitable for neural network input.
Provides one-hot encoding for brick positions.
"""

"""
    flatten(game_state) -> Vector{Float32}

Convert game state to flattened vector suitable for neural network input.

Creates a feature vector with:
- ROWS × COLS binary features: one-hot encoding for each brick position
- 5 continuous features: ball position/velocity, paddle position

# Arguments
- `game_state`: Tuple from get_state() containing (score, ball_cx, ball_cy, ball_vx, ball_vy, paddle_cx, bricks)

# Returns
Vector{Float32} with raw features totaling ROWS * COLUMNS + 5 elements
"""
function flatten(game_state)
    score, ball_cx, ball_cy, ball_vx, ball_vy, paddle_cx, bricks = game_state
    
    # Create output vector with correct size
    grid_size = ROWS * COLS
    output = zeros(Float32, grid_size + 5)

    # Set normalized continuous features
    playable_width = GAME_WIDTH - 2 * WALL_THICKNESS
    playable_height = GAME_HEIGHT - SCORE_AREA_HEIGHT - WALL_THICKNESS
    output[1] = Float32((paddle_cx - WALL_THICKNESS) / playable_width)                 # Paddle X: [0, 1]
    output[2] = Float32((ball_cx - WALL_THICKNESS) / playable_width)                    # Ball X: [0, 1]
    output[3] = Float32((ball_cy - (SCORE_AREA_HEIGHT + WALL_THICKNESS)) / playable_height)  # Ball Y: [0, 1]
    output[4] = Float32(ball_vy)                                                        # Ball vel Y: {-1, 1}
    output[5] = Float32(ball_vx)                                                        # Ball vel X: [-1, 1]
    
    Δ = 6
    # One-hot encoding for each brick position
    for brick in bricks
        # Convert to 0-based grid coordinates
        col = Int((brick.rect.x - WALL_THICKNESS) ÷ BRICK_WIDTH)
        row = Int((brick.rect.y - (SCORE_AREA_HEIGHT + WALL_THICKNESS + 30)) ÷ BRICK_HEIGHT)
        
        # Assert valid indices
        @assert 0 ≤ row < ROWS "Row index $row out of bounds [0, $(ROWS-1)]"
        @assert 0 ≤ col < COLS "Col index $col out of bounds [0, $(COLS-1)]"
        
        output[row * COLS + col +  Δ] = 1.0f0
    end

    return output
end

