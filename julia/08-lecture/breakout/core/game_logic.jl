"""
Core game logic module for Breakout.

Manages game state, physics, collisions, and scoring mechanics.
Designed to work both in interactive mode (GameZero) and simulation mode.
"""
module GameLogic

using Colors
using GameZero
using ..Digits

export score, ball, ball_x, ball_y, ball_vel, paddle, bricks, Brick
export GAME_WIDTH, GAME_HEIGHT, SCORE_AREA_HEIGHT, WALL_THICKNESS
export BALL_SIZE, PADDLE_WIDTH, PADDLE_HEIGHT, MAX_ANGLE
export COLS, ROWS, BRICK_WIDTH, BRICK_HEIGHT
export reset, update_physics, handle_input
export get_game_state, set_simulation_mode

# Game dimensions and layout
const SCORE_AREA_HEIGHT = 20
const WALL_THICKNESS = 10
const COLS = 14  # Brick columns
const ROWS = 6   # Brick rows
const BRICK_WIDTH = 10
const BRICK_HEIGHT = 6
const GAME_WIDTH = COLS * BRICK_WIDTH + 2 * WALL_THICKNESS  # 160 pixels
const GAME_HEIGHT = 210  # Classic Atari height

# Game object sizes
const BALL_SIZE = 2
const PADDLE_WIDTH = 20
const PADDLE_HEIGHT = 3
const MAX_ANGLE = 70  # Maximum ball angle deflection

# Mutable game state variables
score = 0
ball = Rect(GAME_WIDTH / 2, GAME_HEIGHT / 2, BALL_SIZE, BALL_SIZE)
ball_x = GAME_WIDTH / 2.0    # High-precision ball position
ball_y = GAME_HEIGHT / 2.0
ball_vel = (0, 0)
paddle = Rect(GAME_WIDTH / 2 - PADDLE_WIDTH/2, GAME_HEIGHT - 8, PADDLE_WIDTH, PADDLE_HEIGHT)
bricks = []

# Runtime configuration
simulation_mode = false

struct Brick
    brick::Rect
    brick_color
    highlight_color
    points::Int
end

function reset(keep_score=false)
    """Initialize game state with authentic Atari brick layout and ball position"""
    deleteat!(bricks, 1:length(bricks))
    if !keep_score
        global score = 0  # Reset score on game over
    end
    
    # Authentic Atari color palette with point values - 1 row per color
    atari_colors = [
        RGB(0.8, 0.2, 0.2),   # Red (10 points - top row, hardest to reach)
        RGB(0.8, 0.5, 0.1),   # Orange (8 points)
        RGB(0.8, 0.8, 0.3),   # Yellow (6 points)
        RGB(0.3, 0.7, 0.3),   # Green (4 points)
        RGB(0.2, 0.4, 0.8),   # Blue (2 points)
        RGB(0.3, 0.7, 0.7)    # Cyan (1 point - bottom row, easiest to reach)
    ]
    
    # Point values for each row (top to bottom: 10,8,6,4,2,1)
    point_values = [10, 8, 6, 4, 2, 1]
    
    # Create brick wall (in native resolution, positioned below walls)
    brick_start_x = WALL_THICKNESS  # Start after left wall
    brick_start_y = SCORE_AREA_HEIGHT + WALL_THICKNESS + 30  # Start below top wall with spacing
    for x in 1:COLS
        for y in 1:ROWS
            brick_color = atari_colors[y]
            brick_points = point_values[y]
            brick = Brick(
                Rect(brick_start_x + (x-1) * BRICK_WIDTH, brick_start_y + (y-1) * BRICK_HEIGHT, BRICK_WIDTH, BRICK_HEIGHT),
                brick_color, 
                brick_color,
                brick_points
            )
            push!(bricks, brick)
        end
    end

    # Position ball directly below the lowest brick row (native coordinates)
    ball_start_y = brick_start_y + ROWS * BRICK_HEIGHT + 5  # Just below bricks
    global ball_x = GAME_WIDTH / 2.0
    global ball_y = ball_start_y
    ball.center = (ball_x, ball_y)
    global ball_vel = (rand(-60:60), 80)  # Adjusted for native resolution
end

function update_step(dt)
    """Update ball position and handle collisions in native resolution"""
    global ball_x, ball_y, ball_vel
    vx, vy = ball_vel
    
    # Check for ball falling off bottom (game over)
    if ball_y + BALL_SIZE > GAME_HEIGHT
        if !simulation_mode
            reset()  # Auto-restart in interactive mode
        end
        return false  # Game over
    end
    
    # Move ball with floating point precision
    ball_x += vx * dt
    ball_y += vy * dt
    
    # Update Rect for collision detection
    ball.center = (ball_x, ball_y)
    
    # Wall collisions (account for wall thickness and score area)
    if ball_x - BALL_SIZE/2 < WALL_THICKNESS
        vx = -vx
        ball_x = WALL_THICKNESS + BALL_SIZE/2
    elseif ball_x + BALL_SIZE/2 > GAME_WIDTH - WALL_THICKNESS
        vx = -vx
        ball_x = GAME_WIDTH - WALL_THICKNESS - BALL_SIZE/2
    end

    top_wall_position = SCORE_AREA_HEIGHT + WALL_THICKNESS
    if ball_y - BALL_SIZE/2 < top_wall_position
        vy = -vy
        ball_y = top_wall_position + BALL_SIZE/2
    end
    
    # Update ball rect after collision corrections
    ball.center = (ball_x, ball_y)
    
    # Paddle collision
    if collide(ball, paddle)
        # Apply angle based on where ball hits paddle
        vx = (ball_x - paddle.centerx) / (paddle.w / 2) * MAX_ANGLE
        vy = -abs(vy)
    else
        # Brick collisions
        collisions = [collide(ball, brick.brick) for brick in bricks]
        idx = findfirst(x -> x == true, collisions)
        if idx ≠ nothing
            brick = bricks[idx]
            dx = (ball_x - brick.brick.centerx) / BRICK_WIDTH
            dy = (ball_y - brick.brick.centery) / BRICK_HEIGHT
            
            # Determine bounce direction
            if abs(dx) > abs(dy)
                vx = copysign(abs(vx), dx)
            else
                vy = copysign(abs(vy), dy)
            end
            
            # Remove hit brick and update score
            brick_score = bricks[idx].points
            deleteat!(bricks, idx)
            global score += brick_score
            
            # Check if all bricks are destroyed
            if length(bricks) == 0
                global score += 66  # Bonus for clearing all bricks
                reset(true)  # Keep score for next level
                return true  # Continue to next level
            end
        end
    end
    
    ball_vel = (vx, vy)
    return true  # Continue game
end

function update_physics()
    """Main physics update function - returns false on game over"""
    return update_step(1 / 60)
end



function handle_input(action)
    """Common input handling - applies action to paddle"""
    paddle_speed = 1  # Adjusted for native resolution
    
    # Apply action
    paddle.centerx += action * paddle_speed
    
    # Keep paddle within walls (native boundaries)
    if paddle.left < WALL_THICKNESS
        paddle.left = WALL_THICKNESS
    elseif paddle.right > GAME_WIDTH - WALL_THICKNESS
        paddle.right = GAME_WIDTH - WALL_THICKNESS
    end
end


function get_game_state()
    """Get current game state for RL agents"""
    return (score, ball_x, ball_y, ball_vel[1], ball_vel[2], paddle.centerx, bricks)
end

function set_simulation_mode(enabled::Bool)
    """Enable/disable simulation mode (no auto-restart on game over)"""
    global simulation_mode = enabled
end


# Initialize game
reset()

end # module GameLogic
