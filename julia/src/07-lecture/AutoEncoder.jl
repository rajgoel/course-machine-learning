using Random
using Flux
using Plots

# Include MNIST data loading
using ..Lecture03: load_mnist_data

"""
    create_autoencoder(input_dim, hidden_dim, latent_dim)

Create symmetric autoencoder with encoder-decoder architecture.

# Arguments
- `input_dim::Int`: Input dimension (e.g., 784 for MNIST)
- `hidden_dim::Int`: Hidden layer dimension
- `latent_dim::Int`: Latent space dimension (bottleneck)

# Returns
- `Flux.Chain`: Autoencoder model with architecture:
  - Encoder: input → hidden (ReLU) → latent (linear)
  - Decoder: latent → hidden (ReLU) → output (sigmoid)

The model learns to compress input to latent space and reconstruct it.
"""
function create_autoencoder(input_dim::Int, hidden_dim::Int, latent_dim::Int)
    return Chain(
        # Encoder: 784 → hidden → latent
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, latent_dim),
        
        # Decoder: latent → hidden → 784
        Dense(latent_dim, hidden_dim, relu),
        Dense(hidden_dim, input_dim, sigmoid)
    )
end

"""
    train!(model, X_train; epochs=50, η=0.001f0, batch_size=256)

Train autoencoder using reconstruction loss (MSE).

# Arguments
- `model`: Autoencoder model to train
- `X_train`: Training data (features × samples), values in [0,1]
- `epochs::Int`: Number of training epochs (default: 50)
- `η::Float32`: Adam optimizer learning rate (default: 0.001f0)
- `batch_size::Int`: Mini-batch size (default: 256)

# Returns
- `Vector{Float32}`: Training losses per epoch

Trains the autoencoder to minimize MSE between input and reconstruction.
"""
function train!(model, X_train; epochs=50, η=0.001f0, batch_size=256)
    @assert size(X_train, 1) == 784 "Input should be 784 features"
    @assert 0.0 ≤ minimum(X_train) ≤ maximum(X_train) ≤ 1.0 "Data should be [0,1]"
    
    optimizer = Flux.setup(Flux.Adam(η), model)
    data_loader = Flux.DataLoader(X_train, batchsize=batch_size, shuffle=true)
    
    losses = Float32[]
    
    for epoch in 1:epochs
        epoch_losses = Float32[]
        
        for x_batch in data_loader
            loss, grads = Flux.withgradient(model) do m
                Flux.mse(m(x_batch), x_batch)
            end
            
            Flux.update!(optimizer, model, grads[1])
            push!(epoch_losses, loss)
        end
        
        epoch_loss = sum(epoch_losses) / length(epoch_losses)
        push!(losses, epoch_loss)
        
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $(round(epoch_loss, digits=4))")
        end
    end
    
    return losses
end

"""
    extract_encoder(model)

Extract encoder part from trained autoencoder.

# Arguments
- `model`: Trained autoencoder model

# Returns
- `Flux.Chain`: Encoder portion (input → latent space mapping)

Useful for visualizing latent representations or transfer learning.
"""
function extract_encoder(model)
    return Chain(model.layers[1], model.layers[2])  # First two layers
end



