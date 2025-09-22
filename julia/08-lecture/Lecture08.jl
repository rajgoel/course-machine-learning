"""
Lecture 08: Deep Reinforcement Learning

Demonstrates Deep Reinforcement Learning on the Breakout Game
"""

module Lecture08

using Pkg

function __init__()
    # Auto-setup the breakout environment when module loads
    breakout_dir = joinpath(@__DIR__, "breakout")
    if isfile(joinpath(breakout_dir, "Project.toml"))
        original_env = Base.active_project()
        try
            Pkg.activate(breakout_dir)
            Pkg.instantiate()
        finally
            Pkg.activate(original_env)
        end
    end
end

include("breakout/run_game.jl")
include("breakout/run_simulation.jl")

end # module Lecture08
