"""
Policy Agent.

Provides AI agent functionality using trained policy gradient networks.
"""

using Flux
import CommonRLInterface as RL
using Breakout: GameState, flatten

export policy_agent

"""
    policy_agent(game_state::GameState, policy) -> Any

Get action from discrete policy network for Breakout.

# Arguments
- `game_state`: Current Breakout game state
- `policy`: Trained policy network (Flux.Chain with softmax output)

# Returns
- Discrete action from Breakout action space
"""
function policy_agent(game_state::GameState, policy)
    # Get flattened state
    state = flatten(game_state)
    
    # Get action probabilities from policy
    probs = policy(reshape(state, :, 1))[:, 1]
    
    # Sample action from probability distribution  
    action_idx = sample_action(probs)
    
    # Return the actual action (assuming we have access to action space)
    # For now, return the index - this should be mapped to actual actions
    return action_idx
end

