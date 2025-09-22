"""
Random control module for Breakout.

Provides AI agent that moves randomly with probabilistic direction changes.
"""
module Control

using GameZero
using ..GameLogic

export get_action

mutable struct RandomAgent
    current_direction::Int      # Current movement: -1 (left), 0 (stop), 1 (right)
    change_probability::Float64 # Probability of changing direction each step
    
    RandomAgent(change_prob=0.2) = new(0, change_prob)
end

# Global agent instance with 10% direction change probability
const agent = RandomAgent(0.1)

"""
    get_action(g) -> Int

Get paddle movement action using random movement strategy.

Uses probabilistic direction changes to create semi-realistic movement patterns
rather than completely random actions each frame.

# Returns
- `-1`: Move paddle left
- `1`: Move paddle right  
- `0`: No movement
"""
function get_action(g)
    # Probabilistically change movement direction
    if rand() < agent.change_probability
        agent.current_direction = rand([-1, 0, 1])
    end
    
    return agent.current_direction
end

end # module Control
