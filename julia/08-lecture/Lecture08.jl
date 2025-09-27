"""
Lecture 08: Deep Reinforcement Learning

Demonstrates Deep Reinforcement Learning on the Breakout Game
"""

module Lecture08

# Include components
include("breakout/breakout.jl")
using .Breakout
include("imitation_learning/imitation_learning.jl")

export train_imitation, run_imitation, breakout

end # module Lecture08
