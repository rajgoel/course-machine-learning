"""
Digit rendering module for Breakout game score display.

Provides bitmap font rendering for digits 0-9 using 3x5 pixel patterns.
"""
module Digits

export draw_score

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

end # module Digits