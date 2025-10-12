function QNetwork(layers::Vector{Int})
    # Build the model using Flux.Chain
    model_layers = []
    for i in 1:length(layers)-2
        push!(model_layers, Flux.Dense(layers[i], layers[i+1], Flux.relu))
    end
    push!(model_layers, Flux.Dense(layers[end-1], layers[end]))
    return Flux.Chain(model_layers...)
end

