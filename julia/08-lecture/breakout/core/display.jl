module Display

using Colors
using GameZero
using ..GameLogic
using ..Digits

export WIDTH, HEIGHT, BACKGROUND, SCALE_FACTOR
export draw_game, scale_rect

# Display constants  
SCALE_FACTOR = 4
WIDTH = GAME_WIDTH * SCALE_FACTOR   # 640 pixels display width
HEIGHT = GAME_HEIGHT * SCALE_FACTOR # 840 pixels display height
BACKGROUND = colorant"black"

# Scaling utilities
scale_x(x) = x * SCALE_FACTOR
scale_y(y) = y * SCALE_FACTOR
scale_rect(rect) = Rect(scale_x(rect.x), scale_y(rect.y), scale_x(rect.w), scale_y(rect.h))

# All digit rendering now handled by shared DigitRendering module

function draw_game(g::Game)
    """Render all game objects scaled from native resolution"""
    clear()
    
    # Draw gray walls (left, top, right) - positioned below score area
    wall_color = RGB(0.7, 0.7, 0.7)  # Gray walls
    wall_y_start = SCORE_AREA_HEIGHT
    
    # Left wall (from score area down)
    draw(scale_rect(Rect(0, wall_y_start, WALL_THICKNESS, GAME_HEIGHT - SCORE_AREA_HEIGHT)), wall_color, fill=true)
    # Top wall (below score area)
    draw(scale_rect(Rect(0, wall_y_start, GAME_WIDTH, WALL_THICKNESS)), wall_color, fill=true)
    # Right wall (from score area down)
    draw(scale_rect(Rect(GAME_WIDTH - WALL_THICKNESS, wall_y_start, WALL_THICKNESS, GAME_HEIGHT - SCORE_AREA_HEIGHT)), wall_color, fill=true)
    
    # Draw centered score using gray color (doubled size)
    score_digits = length(string(score))
    digit_width = 6  # Double the width (3 * 2)
    spacing = 2      # Double the spacing
    total_score_width = score_digits * digit_width + (score_digits - 1) * spacing
    score_x = (GAME_WIDTH - total_score_width) ÷ 2  # Center horizontally
    score_y = 6  # Vertically centered in larger score area
    # Draw score using shared function
    draw_score(score, score_x, score_y, 2, 4) do px, py
        if px >= 0 && px < GAME_WIDTH && py >= 0 && py < GAME_HEIGHT
            pixel_rect = Rect(px, py, 1, 1)
            draw(scale_rect(pixel_rect), RGB(0.7, 0.7, 0.7), fill=true)
        end
    end
    
    # Draw brick wall (scaled)
    for brick in bricks
        draw(scale_rect(brick.brick), brick.brick_color, fill=true)
    end
    
    # Draw paddle and ball (scaled)
    draw(scale_rect(paddle), colorant"white", fill=true)
    draw(scale_rect(ball), colorant"white", fill=true)
end

end # module Display