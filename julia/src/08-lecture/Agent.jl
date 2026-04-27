using Flux
import CommonRLInterface as RL
using Breakout: GameState, flatten

export agent

"""
    agent(game_state::GameState, model) -> Int

Agent providing paddle control for Breakout.

# Arguments
- `game_state`: Breakout game state
- `model`: Trained DQN network (Flux.Chain)

# Returns
- `-1`: Move paddle left
- `1`: Move paddle right  
- `0`: No movement
"""
function agent(game_state::GameState, model)
    # Get flattened state
    state = flatten(game_state)
    
    # Determine network input size from first layer
    input_size = size(model.layers[1].weight, 2)
    
    if input_size > length(state)
        error("Trained model must take subset of game state as input") 
    end 

    # Get values for all actions
    values = model(reshape(state[1:input_size], :, 1))
    
    # Get the action with highest value
    action_idx = argmax(values[:, 1])
    
    # Convert to Breakout action space [-1, 0, 1]
    actions = [-1, 0, 1]
    return actions[action_idx]
end

