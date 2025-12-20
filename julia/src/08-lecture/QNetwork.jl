using JLD2

function QNetwork(layers::Vector{Int})
    # Build the model using Flux.Chain
    model_layers = []
    for i in 1:length(layers)-2
        push!(model_layers, Flux.Dense(layers[i], layers[i+1], Flux.relu))
    end
    push!(model_layers, Flux.Dense(layers[end-1], layers[end]))
    return Flux.Chain(model_layers...)
end

"""
    save(q_network, filepath::String)

Save a trained DQN network to file using JLD2.

# Arguments
- `q_network`: Trained DQN network (Flux.Chain)
- `filepath`: Path to save file (should end with .jld2)

# Example
```julia
save(q_network, "model.jld2")
```
"""
function save(q_network, filepath::String)
    # Extract architecture from the network structure
    architecture = [size(layer.weight, 2) for layer in q_network.layers]
    push!(architecture, size(q_network.layers[end].weight, 1))
    
    jldsave(filepath; 
        weights=Flux.state(q_network),
        architecture=architecture
    )
    println("Q-network saved to: $filepath")
end

"""
    load(filepath::String) -> Flux.Chain

Load a trained DQN network from file.

# Arguments
- `filepath`: Path to saved network file (.jld2)

# Returns
- Loaded DQN network (Flux.Chain)

# Example
```julia
q_network = load("model.jld2")
```
"""
function load(filepath::String)
    data = JLD2.load(filepath)
    q_network = QNetwork(data["architecture"])
    Flux.loadmodel!(q_network, data["weights"])
    println("Q-network loaded from: $filepath")
    return q_network
end

