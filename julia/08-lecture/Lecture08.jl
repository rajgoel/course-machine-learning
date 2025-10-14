"""
Lecture 08: Deep Reinforcement Learning

Demonstrates Deep Reinforcement Learning on the Breakout Game
"""

module Lecture08

# Include components
include("breakout/breakout.jl")
using .Breakout
include("breakout/interface.jl")
include("breakout/interface_atari.jl")
include("imitation_learning.jl")
include("logger.jl")
include("q_network.jl")
include("replay_buffer.jl")
include("dqn.jl")
include("ppo.jl")
#include("imitation_learning/imitation_learning.jl")
#include("q_learning/agent_base.jl")
#include("q_learning/replay_buffer.jl")
#include("q_learning/q_learning.jl")
#include("q_learning/dqn.jl")
#include("q_learning/dqn_atari.jl")

export train_imitation, run_imitation

end # module Lecture08
