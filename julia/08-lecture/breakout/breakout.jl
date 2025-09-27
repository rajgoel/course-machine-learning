"""
Breakout Game Module

Contains all core functions and control systems for the Breakout game.
"""
module Breakout

# Game runner dependencies
using FileIO, Images, Colors, Statistics

# Include all game files
include("game.jl")
include("draw.jl")
include("flatten.jl")
include("control/keyboard.jl")
include("control/heuristic.jl")
include("control/random.jl")
include("renderer.jl")

# Export main functions that external code needs
export breakout, get_game_state, flatten, render_screenshot

"""
    breakout(control_func=nothing; autorestart=true)

Launch the Breakout game with the specified control function using SDL rendering.

# Arguments
- `control_func`: Function that takes game state and returns action (-1, 0, 1)
  - If nothing, uses default keyboard control
- `autorestart`: Whether to automatically restart the game when ball falls off (default: true)

# Controls
- **Keyboard mode**: Arrow keys or WASD to move paddle, ESC to quit
- **Function mode**: AI/Agent plays automatically, ESC to quit
"""
function breakout(control_func=nothing; autorestart=true)
    if control_func === nothing
        println("🎮 Starting Breakout with SDL rendering, control: keyboard")
        get_control_action = get_keyboard_action
    else
        println("🎮 Starting Breakout with SDL rendering, control: function")
        get_control_action = control_func
    end
    
    # Create window and start game loop
    create_window()
    
    try
        # Reset game to initial state
        reset()
        
        # Main game loop
        running = true
        while running
            # Get current game state
            game_state = get_game_state()
            
            # Process events (with screenshot support)
            running = process_events(game_state)
            if !running
                break
            end
            
            # Get action from control function
            action = get_control_action(game_state)
            
            # Update game
            game_over = !update(action)
            
            # Handle game over based on autorestart setting
            if game_over && autorestart
                reset()
                game_state = get_game_state()  # Get new state after reset
            end
            
            # Render current state
            render_display(game_state)
            
            # Small delay to control frame rate
            sleep(1/60)  # 60 FPS
        end
        
    finally
        close_window()
    end
    
    println("Game ended")
end

end # module Breakout