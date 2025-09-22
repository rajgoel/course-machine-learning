"""
Breakout Game - Main GameZero Interface

Classic Atari Breakout implementation using modular architecture.
Supports multiple control modes: keyboard, random AI, and heuristic AI.
"""

using GameZero
using FileIO, Images

# Load control module based on environment variable
const CONTROL_MODULE = get(ENV, "CONTROL_MODULE", "control/keyboard.jl")

# Load game modules using GameZero's module system
game_include("core/digits.jl")
game_include("core/game_logic.jl")
game_include("core/screenshot.jl")
game_include("core/display.jl") 
game_include(CONTROL_MODULE)

# Import all required modules
using .Digits
using .GameLogic
using .Screenshot
using .Display
using .Control

# GameZero callback functions
function draw(g::Game)
    draw_game(g)
end

function update(g::Game)
    action = get_action(g)
    handle_input(action)
    update_physics()
end

function on_key_down(g::Game, key, scancode)
    if key == Keys.ESCAPE
        GameZero.playing[] = false
    elseif key == Keys.P
        # Save current game state as image
        game_state = get_game_state()
        pixels = screenshot(game_state)
        filename = "breakout_state_$(round(Int, time())).png"
        save(filename, Gray.(pixels))
        println("Pixel state saved as $filename")
    end
end
