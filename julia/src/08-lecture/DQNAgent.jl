using Flux
import CommonRLInterface as RL
using Breakout: GameState, flatten

export dqn_action

"""
    dqn_agent(game_state::GameState, q_network) -> Int

DQN agent providing paddle control for Breakout.

# Arguments
- `game_state`: Breakout game state
- `q_network`: Trained DQN network (Flux.Chain)

# Returns
- `-1`: Move paddle left
- `1`: Move paddle right  
- `0`: No movement
"""
function dqn_agent(game_state::GameState, q_network)
    # Get flattened state
    state = flatten(game_state)
    
    # Get Q-values for all actions
    q_values = q_network(reshape(state, :, 1))
    
    # Get the action with highest Q-value
    action_idx = argmax(q_values[:, 1])
    
    # Convert to Breakout action space [-1, 0, 1]
    actions = [-1, 0, 1]
    return actions[action_idx]
end

