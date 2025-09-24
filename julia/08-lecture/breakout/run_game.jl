"""
Breakout Game Runner

Provides a function interface to launch the Breakout game with different control modes.
Handles environment management and dependency loading automatically.
"""

using GameZero
using FileIO, Images, Colors, Statistics
import Pkg

"""
    breakout(control_file="08-lecture/breakout/control/keyboard.jl")

Launch the Breakout game with the specified control module.

# Arguments
- `control_file`: Path to control module relative to current working directory
  - Built-in options: "08-lecture/breakout/control/keyboard.jl", "08-lecture/breakout/control/random.jl", "08-lecture/breakout/control/heuristic.jl"
  - Custom files: Any path relative to pwd() or absolute path

# Controls
- **Keyboard mode**: Arrow keys or WASD to move paddle, ESC to quit, P to save screenshot
- **Random/Heuristic modes**: AI plays automatically, ESC to quit, P to save screenshot
"""
function breakout(control_file="08-lecture/breakout/control/keyboard.jl")
    println("🎮 Starting Breakout with control: $control_file")
    
    # Switch to breakout environment temporarily
    original_env = Base.active_project()
    original_dir = pwd()
    breakout_dir = @__DIR__
    
    # Resolve control file path relative to current working directory
    control_path = isabspath(control_file) ? control_file : abspath(control_file)
    
    try
        # Activate breakout environment and ensure packages are installed
        Pkg.activate(breakout_dir)
        if !isfile(joinpath(breakout_dir, "Manifest.toml"))
            println("Installing breakout dependencies...")
            Pkg.instantiate()
        end
        
        cd(breakout_dir)
        
        # Create temporary game file with control path embedded
        temp_game_content = """
        using GameZero
        using FileIO, Images

        # Load game modules
        game_include("core/digits.jl")
        game_include("core/game_logic.jl")
        game_include("core/screenshot.jl")
        game_include("core/display.jl")
        game_include("core/flatten.jl")
        game_include("$control_path")

        # Import modules
        using .Digits
        using .GameLogic
        using .Screenshot
        using .Display
        using .Flatten
        using .Control

        # GameZero callbacks
        function draw(g)
            draw_game(g)
        end

        function update(g)
            action = get_action(g)
            handle_input(action)
            update_physics()
        end

        function on_key_down(g, key, scancode)
            if key == GameZero.Keys.ESCAPE
                GameZero.playing[] = false
            elseif key == GameZero.Keys.P
                game_state = get_game_state()
                pixels = screenshot(game_state)
                filename = "breakout_state_\$(round(Int, time())).png"
                save(filename, Gray.(pixels))
                println("Pixel state saved as \$filename")
            end
        end
        """
        
        # Write and run temporary game file
        temp_file = joinpath(breakout_dir, "temp_game.jl")
        write(temp_file, temp_game_content)
        
        try
            GameZero.rungame(temp_file)
        finally
            rm(temp_file, force=true)
        end
        
    finally
        # Switch back to original environment and directory
        Pkg.activate(original_env)
        cd(original_dir)
    end
end
