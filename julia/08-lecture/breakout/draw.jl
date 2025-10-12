"""
Shared drawing functions for both SDL rendering and headless screenshot generation.

This module provides a common drawing interface that can output to different targets
(SDL renderer or pixel arrays) by accepting a pixel drawing callback function.
Includes bitmap font rendering for score display.
"""

using Colors

# 3x5 pixel bitmap patterns for digits 0-9
const DIGIT_PATTERNS = [
    [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1],  # 0
    [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1],  # 1 
    [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1],  # 2
    [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1],  # 3
    [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1],  # 4
    [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1],  # 5
    [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1],  # 6
    [1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1],  # 7
    [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1],  # 8
    [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1]   # 9
]

"""
    draw_digit(pixel_func, digit, x_pos, y_pos, scale=1)

Draw a single digit using the provided pixel callback function.

# Arguments
- `pixel_func`: Function(x, y) called for each pixel to draw
- `digit`: Digit to draw (0-9)
- `x_pos, y_pos`: Position to draw at
- `scale`: Scaling factor (1 = 3x5, 2 = 6x10, etc.)
"""
function draw_digit(pixel_func, digit, x_pos, y_pos, scale=1)
    digit < 0 || digit > 9 && return
    
    pattern = DIGIT_PATTERNS[digit + 1]
    
    for row in 1:5, col in 1:3
        pixel_idx = (row-1) * 3 + col
        pattern[pixel_idx] == 1 || continue
        
        # Draw scaled pixel blocks
        for dy in 0:(scale-1), dx in 0:(scale-1)
            y_coord = y_pos + (row-1) * scale + dy
            x_coord = x_pos + (col-1) * scale + dx
            pixel_func(x_coord, y_coord)
        end
    end
end

"""
    draw_hash(pixel_func, x_pos, y_pos, scale=2)

Draw a hash '#' symbol using the provided pixel callback function.

# Arguments
- `pixel_func`: Function(x, y) called for each pixel to draw
- `x_pos, y_pos`: Position to draw at
- `scale`: Scaling factor
"""
function draw_hash(pixel_func, x_pos, y_pos, scale=2)
    # 5x5 pattern for '#' symbol
    hash_pattern = [
        0,1,0,1,0,
        1,1,1,1,1,
        0,1,0,1,0,
        1,1,1,1,1,
        0,1,0,1,0
    ]
    
    for row in 1:5, col in 1:5
        pixel_idx = (row-1) * 5 + col
        hash_pattern[pixel_idx] == 1 || continue
        
        # Draw scaled pixel blocks
        for dy in 0:(scale-1), dx in 0:(scale-1)
            y_coord = y_pos + (row-1) * scale + dy
            x_coord = x_pos + (col-1) * scale + dx
            pixel_func(x_coord, y_coord)
        end
    end
end

"""
    draw_score(pixel_callback, score_value, x, y, scale=2, spacing=4)

Draw a multi-digit score using bitmap font rendering.

# Arguments
- `pixel_callback`: Function(x, y) called for each pixel to draw
- `score_value`: The score number to display
- `x, y`: Position to draw at
- `scale`: Scaling factor for digits
- `spacing`: Horizontal spacing between digits
"""
function draw_score(pixel_callback, score_value, x, y, scale=2, spacing=4)
    score_str = string(score_value)
    
    for (i, char) in enumerate(score_str)
        digit = parse(Int, char)
        0 ≤ digit ≤ 9 || continue
        
        x_pos = x + (i-1) * spacing * scale
        draw_digit(pixel_callback, digit, x_pos, y, scale)
    end
end

"""
    draw_game(game_state, draw_pixel, current_game)

Draw the complete game state using the provided pixel drawing function.

# Arguments
- `game_state`: Current game state tuple (score, ball_cx, ball_cy, ball_vx, ball_vy, paddle_cx, bricks)
- `draw_pixel(x, y, color)`: Callback function to draw a pixel at (x,y) with given color
- `current_game`: Current game counter to display

The color parameter format depends on the target:
- For SDL: (r, g, b, a) as UInt8 values 0-255
- For screenshots: grayscale Float64 value 0.0-1.0
"""
function draw_game(game_state, draw_pixel, current_game)
    score, ball_cx, ball_cy, ball_vx, ball_vy, paddle_cx, bricks = game_state
    
    # Draw score digits (centered in left half)
    score_x_start = GAME_WIDTH ÷ 4 - length(string(score)) * 2
    draw_score(score, score_x_start, 6, 2, 4) do px, py
        if px >= 1 && px <= GAME_WIDTH && py >= 1 && py <= GAME_HEIGHT
            draw_pixel(px, py, (128, 128, 128, 255))  # Gray color for score
        end
    end
    
    # Draw game counter (centered in right half) with '#' prefix
    hash_width = 5 * 2  # 5x5 pattern with scale=2
    number_width = length(string(current_game)) * 4 * 2  # digits with scale=2 and spacing=4
    total_width = hash_width + 2 + number_width  # hash + gap + number
    game_counter_x_start = 3 * GAME_WIDTH ÷ 4 - total_width ÷ 2
    
    # Draw '#' symbol
    draw_hash(game_counter_x_start, 6, 2) do px, py
        if px >= 1 && px <= GAME_WIDTH && py >= 1 && py <= GAME_HEIGHT
            draw_pixel(px, py, (128, 128, 128, 255))  # Gray color for hash
        end
    end
    
    # Draw the number after the '#' symbol
    number_x_start = game_counter_x_start + hash_width + 2
    draw_score(current_game, number_x_start, 6, 2, 4) do px, py
        if px >= 1 && px <= GAME_WIDTH && py >= 1 && py <= GAME_HEIGHT
            draw_pixel(px, py, (128, 128, 128, 255))  # Gray color for game counter
        end
    end
    
    # Draw walls
    wall_thickness = WALL_THICKNESS
    wall_start_y = SCORE_AREA_HEIGHT
    
    # Top wall
    for x in 1:GAME_WIDTH
        for y in wall_start_y+1:wall_start_y+wall_thickness
            draw_pixel(x, y, (128, 128, 128, 255))  # Gray walls
        end
    end
    
    # Left wall (below score area)
    for x in 1:wall_thickness
        for y in wall_start_y+1:GAME_HEIGHT
            draw_pixel(x, y, (128, 128, 128, 255))
        end
    end
    
    # Right wall (below score area)
    for x in GAME_WIDTH-wall_thickness+1:GAME_WIDTH
        for y in wall_start_y+1:GAME_HEIGHT
            draw_pixel(x, y, (128, 128, 128, 255))
        end
    end
    
    # Draw bricks
    for brick in bricks
        color = brick.color
        r = round(UInt8, color.r * 255)
        g = round(UInt8, color.g * 255)
        b = round(UInt8, color.b * 255)
        
        x_start = max(WALL_THICKNESS+1, Int(floor(brick.rect.x)))
        x_end = min(GAME_WIDTH-WALL_THICKNESS+1, Int(ceil(brick.rect.x + brick.rect.w)))
        y_start = max(1, Int(floor(brick.rect.y)))
        y_end = min(GAME_HEIGHT, Int(ceil(brick.rect.y + brick.rect.h)))
        
        for x in x_start:x_end-1
            for y in y_start:y_end-1
                draw_pixel(x, y, (r, g, b, 255))
            end
        end
    end
    
    # Draw paddle (white)
    paddle_x_start = max(WALL_THICKNESS+1, Int(floor(paddle_cx - PADDLE_WIDTH/2)))
    paddle_x_end = min(GAME_WIDTH-WALL_THICKNESS+1, Int(ceil(paddle_cx + PADDLE_WIDTH/2)))
    paddle_y_start = Int(GAME_HEIGHT - 8)
    paddle_y_end = min(GAME_HEIGHT, paddle_y_start + PADDLE_HEIGHT)
    
    for x in paddle_x_start:paddle_x_end-1
        for y in paddle_y_start:paddle_y_end-1
            draw_pixel(x, y, (255, 255, 255, 255))  # White paddle
        end
    end
    
    # Draw ball (white)
    ball_center_x = Int(round(ball_cx))
    ball_center_y = Int(round(ball_cy))
    half_size = BALL_SIZE ÷ 2
    
    left = ball_center_x - half_size
    right = ball_center_x + half_size
    top = ball_center_y - half_size
    bottom = ball_center_y + half_size
    
    for x in left:right
        for y in top:bottom
            if x >= 1 && x <= GAME_WIDTH && y >= 1 && y <= GAME_HEIGHT
                draw_pixel(x, y, (255, 255, 255, 255))  # White ball
            end
        end
    end
end