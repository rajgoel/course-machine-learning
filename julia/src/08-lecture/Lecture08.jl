"""
    Lecture07

Deep Q-learning on the Breakout game

# Available Functions

- `demo()`: Train agent for the Breakout game using DQN
"""

module Lecture08

# Include components
using Breakout
include("Logger.jl")
include("QNetwork.jl")
include("ReplayBuffer.jl")
include("DQN.jl")
include("Demo.jl")

export demo

end # module Lecture08
