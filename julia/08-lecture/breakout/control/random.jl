"""
Random control functions for Breakout.

Provides AI agent that moves randomly with probabilistic direction changes.
"""

mutable struct RandomAgent
    current_direction::Int      # Current movement: -1 (left), 0 (stop), 1 (right)
    change_probability::Float64 # Probability of changing direction each step
    
    RandomAgent(change_prob=0.2) = new(0, change_prob)
end

# Global agent instance with 10% direction change probability
const agent = RandomAgent(0.1)

"""
    get_random_action(game_state) -> Int

Get paddle movement action using random movement strategy.

Uses probabilistic direction changes to create semi-realistic movement patterns
rather than completely random actions each frame.

# Arguments
- `game_state`: Current game state (ignored for random movement)

# Returns
- `-1`: Move paddle left
- `1`: Move paddle right  
- `0`: No movement
"""
function get_random_action(game_state)
    # Probabilistically change movement direction
    if rand() < agent.change_probability
        agent.current_direction = rand([-1, 0, 1])
    end
    
    return agent.current_direction
end

