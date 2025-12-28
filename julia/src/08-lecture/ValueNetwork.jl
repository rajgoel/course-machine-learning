using JLD2

function ValueNetwork(layers::Vector{Int})
    # Build the model using Flux.Chain
    model_layers = []
    for i in 1:length(layers)-2
        push!(model_layers, Flux.Dense(layers[i], layers[i+1], Flux.relu))
    end
    push!(model_layers, Flux.Dense(layers[end-1], layers[end]))
    return Flux.Chain(model_layers...)
end

"""
    save(network, filepath::String)

Save a trained DQN network to file using JLD2.

# Arguments
- `network`: Trained network (Flux.Chain)
- `filepath`: Path to save file (should end with .jld2)

# Example
```julia
save(network, "model.jld2")
```
"""
function save(network, filepath::String)
    # Extract architecture from the network structure
    architecture = [size(layer.weight, 2) for layer in network.layers]
    push!(architecture, size(network.layers[end].weight, 1))
    
    jldsave(filepath; 
        weights=Flux.state(network),
        architecture=architecture
    )
    println("Network saved to: $filepath")
end

"""
    load(filepath::String) -> Flux.Chain

Load a trained DQN network from file.

# Arguments
- `filepath`: Path to saved network file (.jld2)

# Returns
- Loaded network (Flux.Chain)

# Example
```julia
network = load("model.jld2")
```
"""
function load(filepath::String)
    data = JLD2.load(filepath)
    network = ValueNetwork(data["architecture"])
    Flux.loadmodel!(network, data["weights"])
    println("Network loaded from: $filepath")
    return network
end

