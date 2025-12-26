using Flux
using JLD2
using Random

"""
    PolicyNetwork(layers::Vector{Int})

Create a policy network for discrete actions with softmax output.

Outputs action probabilities using softmax for discrete action spaces.

# Arguments
- `layers`: Network architecture (e.g., [state_dim, 64, 32, action_dim])

# Returns
- Policy network (Flux.Chain) with softmax output for discrete actions

# Example
```julia
# For Breakout: 2 state features, 3 discrete actions
policy = PolicyNetwork([2, 64, 32, 3])
probs = Flux.softmax(policy(state)[:, 1])
```
"""
function PolicyNetwork(layers::Vector{Int})
    model_layers = []
    
    # Hidden layers with ReLU activation
    for i in 1:length(layers)-2
        push!(model_layers, Flux.Dense(layers[i], layers[i+1], Flux.relu))
    end
    
    # Output layer: linear activation for logits (softmax applied externally)
    push!(model_layers, Flux.Dense(layers[end-1], layers[end]))
    
    return Flux.Chain(model_layers...)
end

"""
    save(policy, filepath::String)

Save a trained policy network to file using JLD2.

# Arguments
- `policy`: Trained policy network (Flux.Chain)
- `filepath`: Path to save file (should end with .jld2)

# Example
```julia
save(policy, "my_policy.jld2")
```
"""
function save(policy, filepath::String)
    # Extract architecture from the network structure
    architecture = [size(layer.weight, 2) for layer in policy.layers]
    push!(architecture, size(policy.layers[end].weight, 1))
    
    jldsave(filepath; 
        weights=Flux.state(policy),
        architecture=architecture,
        network_type="PolicyNetwork"
    )
    println("Policy network saved to: $filepath")
end

"""
    load(filepath::String) -> Flux.Chain

Load a trained policy network from file.

# Arguments
- `filepath`: Path to saved network file (.jld2)

# Returns
- Loaded policy network (Flux.Chain)

# Example
```julia
policy = load("my_policy.jld2")
```
"""
function load(filepath::String)
    data = JLD2.load(filepath)
    policy = PolicyNetwork(data["architecture"])
    Flux.loadmodel!(policy, data["weights"])
    println("Policy network loaded from: $filepath")
    return policy
end

"""
Sample action from probability distribution.
"""
function sample_action(probs)
    cumsum_probs = cumsum(probs)
    r = rand()
    action_idx = findfirst(p -> p >= r, cumsum_probs)
    if action_idx === nothing  # Numerical precision fix
        action_idx = length(probs)
    end
    return action_idx
end
