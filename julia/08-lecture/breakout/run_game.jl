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
        
        ENV["CONTROL_MODULE"] = control_path
        breakout_file = joinpath(breakout_dir, "breakout.jl")
        rungame(breakout_file)
    finally
        # Switch back to original environment and directory
        Pkg.activate(original_env)
        cd(original_dir)
    end
end
