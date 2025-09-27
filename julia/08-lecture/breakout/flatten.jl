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
- `game_state`: Tuple from get_game_state() containing (score, ball_x, ball_y, ball_vx, ball_vy, paddle_x, bricks)

# Returns
Vector{Float32} with raw features totaling ROWS * COLUMNS + 5 elements
"""
function flatten(game_state)
    score, ball_x, ball_y, ball_vx, ball_vy, paddle_x, bricks = game_state
    
    # Create output vector with correct size
    grid_size = ROWS * COLS
    output = zeros(Float32, grid_size + 5)
    
    # One-hot encoding for each brick position
    for brick in bricks
        # Convert to 0-based grid coordinates
        col = Int((brick.rect.x - WALL_THICKNESS) ÷ BRICK_WIDTH)
        row = Int((brick.rect.y - (SCORE_AREA_HEIGHT + WALL_THICKNESS + 30)) ÷ BRICK_HEIGHT)
        
        # Assert valid indices
        @assert 0 ≤ row < ROWS "Row index $row out of bounds [0, $(ROWS-1)]"
        @assert 0 ≤ col < COLS "Col index $col out of bounds [0, $(COLS-1)]"
        
        output[row * COLS + col + 1] = 1.0f0
    end

    # Set continuous features
    output[grid_size + 1] = Float32(ball_x)
    output[grid_size + 2] = Float32(ball_y)
    output[grid_size + 3] = Float32(ball_vx)
    output[grid_size + 4] = Float32(ball_vy)
    output[grid_size + 5] = Float32(paddle_x)
    
    return output
end

