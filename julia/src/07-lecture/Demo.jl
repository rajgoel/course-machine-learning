
"""
    demo(; train_size=5000, epochs=50, hidden_dim=128, latent_dim=2, display=true)

Complete MNIST autoencoder demo with 2D visualization.

# Arguments
- `train_size::Int`: Number of training samples (default: 5000)
- `epochs::Int`: Number of training epochs (default: 50)
- `hidden_dim::Int`: Hidden layer dimension (default: 128)
- `latent_dim::Int`: Latent space dimension (default: 2)
- `display::Bool`: Whether to display latent space plot (default: true)

# Returns
- `Tuple`: (X_test, model, encoder, plot)
  - `X_test::Array{Float32, 2}`: Flattened test images (784 × 1000)
  - `Y_test::Int`: Labels
  - `autoencoder`: Trained autoencoder model
  - `encoder`: Encoder part of the autoencoder  
"""
function demo(; train_size=5000, epochs=50, hidden_dim=128, latent_dim=2, display=true)
    println("MNIST Autoencoder Demo")
    println("="^50)
    
    # Load data
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = load_mnist_data(train_size, 1000)
    
    # Flatten for dense layers
    X_train = reshape(X_train_raw, 784, train_size)
    Y_train = Y_train_raw
    X_test = reshape(X_test_raw, 784, 1000)
    Y_test = Y_test_raw
    
    # Extract labels for visualization
    Y_test_labels = [findfirst(Bool.(Y_test[:, i])) - 1 for i in 1:1000]
    
    println("Data: train=$(size(X_train)), test=$(size(X_test))")
    
    # Create and train autoencoder
    autoencoder = create_autoencoder(784, hidden_dim, latent_dim)
    println("Architecture: 784 → $hidden_dim → $latent_dim → $hidden_dim → 784")
    
    losses = train!(autoencoder, X_train; epochs=epochs)
    
    println("Training complete! Final loss: $(round(losses[end], digits=4))")

    encoder = extract_encoder(autoencoder)

    # Create 2D visualization
    if display
      display_latent_space(encoder, X_test, Y_test_labels)
    end 
    return X_test, Y_test_labels, autoencoder, encoder
end


"""
    display_latent_space(encoder, X, Y)

Display 2D latent space visualization colored by digit class.

# Arguments
- `encoder`: Trained encoder model
- `X::Array{Float32, 2}`: Flattened images (784 × n)
- `Y::Vector{Int}`: Digit labels (0-9)

# Returns
- Plot object
"""
function display_latent_space(encoder, X, Y)
    # Encode to 2D
    latent = encoder(X)
    x_coords = latent[1, :]
    y_coords = latent[2, :]
    
    # Colors for each digit
    colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
    
    # Create plot
    p = plot(legend=:outertopright, grid=false, showaxis=false, xticks=false, yticks=false)
    
    for digit in 0:9
        mask = Y .== digit
        if sum(mask) > 0
            scatter!(p, x_coords[mask], y_coords[mask], 
                    color=colors[digit+1], alpha=0.7, markersize=3,
                    label="Digit $digit")
        end
    end
    
    display(p)
    return p
end

"""
    display_reconstruction(model, image)

Display original vs reconstructed image comparison.

# Arguments
- `model`: Trained autoencoder model
- `image::Array{Float32, 1}`: Flattened image vector (784 elements)

# Returns
- `Tuple`: (plot, error)
"""
function display_reconstruction(model, image::Array{Float32, 1})
    # Reshape to 2D for model input
    image_flat = reshape(image, 784, 1)
    
    # Calculate reconstruction
    reconstructed = model(image_flat)
    error = sum((image_flat .- reconstructed).^2)
    
    # Reshape to 28×28 for visualization
    original_img = reshape(image, 28, 28)
    reconstructed_img = reshape(reconstructed[:, 1], 28, 28)
    
    # Original
    p1 = heatmap(reverse(original_img, dims=1), color=:grays, aspect_ratio=:equal, 
                showaxis=false, xticks=false, yticks=false, 
                colorbar=false, legend=false, title="Original",
                margin=1Plots.mm, top_margin=-3Plots.mm)
    
    # Reconstruction  
    p2 = heatmap(reverse(reconstructed_img, dims=1), color=:grays, aspect_ratio=:equal,
                showaxis=false, xticks=false, yticks=false,
                colorbar=false, legend=false, title="Reconstruction\n(Error: $(round(error, digits=4)))",
                margin=1Plots.mm, top_margin=-3Plots.mm)
    
    # Combine plots
    plot = Plots.plot(p1, p2, layout=(1, 2), margin=1Plots.mm)
    
    display(plot)
    
    return plot, error
end


