module Screenshot

using Colors
using GameZero: Rect
using ..GameLogic: GAME_WIDTH, GAME_HEIGHT, SCORE_AREA_HEIGHT, WALL_THICKNESS, BALL_SIZE, PADDLE_WIDTH, PADDLE_HEIGHT
using ..Digits

export screenshot

function screenshot(game_state)
    """Render game state as grayscale pixel array for RL agents
    
    Args:
        game_state: Tuple of (score, ball_x, ball_y, ball_vx, ball_vy, paddle_x, bricks)
    
    Returns:
        2D array (GAME_HEIGHT × GAME_WIDTH) with grayscale values 0.0-1.0
    """
    score, ball_x, ball_y, ball_vx, ball_vy, paddle_x, bricks = game_state
    
    # Initialize empty game field
    pixels = zeros(Float64, GAME_HEIGHT, GAME_WIDTH)
    
    # Score area has black background (like the actual game)
    
    # Draw score digits using shared function
    score_x_start = GAME_WIDTH ÷ 2 - length(string(score)) * 4
    draw_score((px, py) -> begin
        if py >= 1 && py <= size(pixels, 1) && px >= 1 && px <= size(pixels, 2)
            pixels[py, px] = 0.5
        end
    end, score, score_x_start, 6, 2, 4)
    
    # Draw walls (gray)
    # Top wall
    pixels[SCORE_AREA_HEIGHT+1:SCORE_AREA_HEIGHT+WALL_THICKNESS, :] .= 0.5
    # Left wall (only below score area)
    pixels[SCORE_AREA_HEIGHT+1:end, 1:WALL_THICKNESS] .= 0.5
    # Right wall (only below score area)
    pixels[SCORE_AREA_HEIGHT+1:end, end-WALL_THICKNESS+1:end] .= 0.5
    
    # Draw bricks (convert original colors to grayscale)
    for brick in bricks
        x_start = max(WALL_THICKNESS + 1, Int(floor(brick.brick.left)))
        x_end = min(GAME_WIDTH, Int(ceil(brick.brick.right)))
        y_start = max(1, Int(floor(brick.brick.top)))
        y_end = min(GAME_HEIGHT, Int(floor(brick.brick.bottom)) - 1)
        
        if x_start <= x_end && y_start <= y_end
            # Convert RGB to grayscale using luminance formula
            color = brick.brick_color
            gray_value = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b
            pixels[y_start:y_end, x_start:x_end] .= gray_value
        end
    end
    
    # Draw paddle (white)
    paddle_left = paddle_x - PADDLE_WIDTH/2
    paddle_top = GAME_HEIGHT - 8
    x_start = max(1, Int(round(paddle_left)))
    x_end = min(GAME_WIDTH, Int(round(paddle_left + PADDLE_WIDTH)))
    y_start = max(1, Int(round(paddle_top)))
    y_end = min(GAME_HEIGHT, Int(round(paddle_top + PADDLE_HEIGHT)))
    
    if x_start <= x_end && y_start <= y_end
        pixels[y_start:y_end, x_start:x_end] .= 1.0
    end
    
    # Draw ball (white) - using actual BALL_SIZE
    ball_x_start = max(1, Int(round(ball_x - BALL_SIZE/2)))
    ball_x_end = min(GAME_WIDTH, Int(round(ball_x + BALL_SIZE/2)))
    ball_y_start = max(1, Int(round(ball_y - BALL_SIZE/2)))
    ball_y_end = min(GAME_HEIGHT, Int(round(ball_y + BALL_SIZE/2)))
    
    if ball_x_start <= ball_x_end && ball_y_start <= ball_y_end
        pixels[ball_y_start:ball_y_end, ball_x_start:ball_x_end] .= 1.0
    end
    
    return pixels
end


end # module Screenshot