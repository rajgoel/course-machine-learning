using Flux
using Statistics
using LinearAlgebra
using SparseArrays

"""
    compute_final_embeddings(model)

Compute node embeddings using GCN message passing layers.

# Arguments
- `model::NamedTuple`: GCN model with initial_embeddings, gcn_layers, and Â

# Returns
- `Matrix{Float32}`: Final node embeddings (n_nodes × embed_dim)
"""
function compute_final_embeddings(model)
    # Get adjacency matrix and initial embeddings
    Â = model.Â
    H = model.initial_embeddings.weight'
    
    # Compute embeddings of each GCN layer: H^(l+1) = φ(Â H^l W^l)
    for layer in model.gcn_layers
        # Extract weights from Flux layer
        W = layer.weight' 
        
        # Neighbor aggregation using linear activation φ(Z) = Z
        H = Â * H * W
    end
    
    return H
end

"""
    similarities(E₁, E₂)

Compute cosine similarities between two embedding matrices.

# Arguments
- `E₁::Matrix{Float32}`: First set of embeddings (batch_size, embed_dim)
- `E₂::Matrix{Float32}`: Second set of embeddings (batch_size, embed_dim)

# Returns
- `Vector{Float32}`: Cosine similarities
"""
function similarities(E₁, E₂)
    ε = 1f-8  # small constant for numerical stability   
    # Compute norms: ‖E‖ = √(∑ E² + ε)
    norm_E₁ = sqrt.(sum(E₁.^2, dims=2) .+ ε)
    norm_E₂ = sqrt.(sum(E₂.^2, dims=2) .+ ε)
    
    # Cosine similarity: (E₁/‖E₁‖) ⋅ (E₂/‖E₂‖)
    return vec(sum(E₁ ./ norm_E₁ .* E₂ ./ norm_E₂, dims=2))
end

"""
    train!(model, graph; epochs=100, η=0.001, batch_size=1024)

Train GCN model to predict edge weights using mini-batch gradient descent.

# Arguments
- `model::NamedTuple`: GCN model with initial_embeddings, gcn_layers, and Â
- `graph::Graph`: Training graph with edges using consecutive indices
- `epochs::Int`: Number of training epochs (default: 50)
- `η::Float32`: Learning rate for Adam optimizer (default: 0.001)
- `batch_size::Int`: Mini-batch size for SGD (default: 1024)

# Returns
- `Vector{Float32}`: Training losses per epoch
"""
function train!(model, graph::Graph; epochs=100, η=0.001, batch_size=1024)
    origin_indices = [edge.origin_index for edge in graph.edges]
    destination_indices = [edge.destination_index for edge in graph.edges]
    weights = [edge.weight for edge in graph.edges]
    
    # Create optimizer
    optimizer = Flux.setup(Flux.Adam(η), [model.initial_embeddings, model.gcn_layers...])
    
    losses = Float32[]
    
    for epoch in 1:epochs
        epoch_losses = Float32[]
        
        for start_idx in 1:batch_size:length(graph.edges)
            end_idx = min(start_idx + batch_size - 1, length(graph.edges))
            batch_indices = start_idx:end_idx
            
            batch_origins = origin_indices[batch_indices]
            batch_destinations = destination_indices[batch_indices]
            batch_weights = weights[batch_indices]
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient([model.initial_embeddings, model.gcn_layers...]) do (initial_embeddings, gcn_layers...)
                # Compute full GCN embeddings
                current_model = (initial_embeddings=initial_embeddings, gcn_layers=gcn_layers, Â=model.Â)
                final_embeddings = compute_final_embeddings(current_model)
                
                # Use similarity of final embeddings
                batch_predictions = similarities(final_embeddings[batch_origins, :], final_embeddings[batch_destinations, :])
                
                # MSE loss
                Flux.mse(batch_predictions, batch_weights)
            end
            
            # Update parameters
            Flux.update!(optimizer, [model.initial_embeddings, model.gcn_layers...], grads[1])
            push!(epoch_losses, loss)
        end
        
        epoch_loss = mean(epoch_losses)
        push!(losses, epoch_loss)
        
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $(round(epoch_loss, digits=4))")
        end
    end
    
    return losses
end

"""
    GCN(graph::Graph; epochs=50, embedding_sizes=[64, 32], η=0.001, batch_size=1024)

Create and train a Graph Convolutional Network.

# Arguments
- `graph::Graph`: Complete graph with nodes and edges using consecutive indices
- `epochs::Int`: Number of training epochs (default: 50)
- `embedding_sizes::Vector{Int}`: Embedding dimensions for each layer (default: [64, 32])
- `η::Float64`: Learning rate for Adam optimizer (default: 0.001)
- `batch_size::Int`: Mini-batch size for SGD (default: 1024)

# Returns
- `Tuple`: (model, embeddings, losses)
"""
function GCN(graph::Graph; epochs=50, embedding_sizes=[128, 64], η=0.001, batch_size=1024)
    A = create_adjacency_matrix(graph; use_weights=true)
    # Add self-loops for GCN
    I = spdiagm(ones(Float32, size(A, 1)))
    
    # Apply GCN normalization: D⁻¹ᐟ² (A + I) D⁻¹ᐟ²
    degrees = Float32.(vec(sum(abs.(A), dims=2))) .+ 1.0f0 # Sum absolute weights
    D⁻¹ᐟ² = spdiagm(degrees.^(-0.5f0))
    Â = D⁻¹ᐟ² * (A + I) * D⁻¹ᐟ²
    
    println("Graph statistics:")
    println("  Nodes: $(length(graph.nodes))")
    println("  Edges: $(length(graph.edges))")
    println("  Edge weight range: $(extrema([e.weight for e in graph.edges]))")
    
    # Create GCN model
    initial_embeddings = Flux.Embedding(length(graph.nodes), embedding_sizes[1])
    gcn_layers = [Flux.Dense(embedding_sizes[i], embedding_sizes[i+1]; bias=false) for i in 1:length(embedding_sizes)-1]
    model = (initial_embeddings=initial_embeddings, gcn_layers=gcn_layers, embedding_sizes=embedding_sizes, Â=Â)
    println("Created GCN model: $(embedding_sizes) embedding sizes, $(length(embedding_sizes)) total layers")
    
    # Train
    println("Training...")
    losses = train!(model, graph, epochs=epochs, η=η, batch_size=batch_size)
    
    # Compute final embeddings
    embeddings = compute_final_embeddings(model)
    println("Final embeddings shape: $(size(embeddings))")
    
    return model, embeddings, losses
end
