"""
LeNet-5 CNN Architecture Implementation using Flux.jl

Provides FluxLeNet5 struct following the same API pattern as FluxDNN.
This implementation follows the original LeNet-5 architecture (but uses ϕ = ReLU as default instead of ϕ = sigmoid):
- Conv2D (6 filters, 5x5) + ϕ + AvgPool (2x2)
- Conv2D (16 filters, 5x5) + ϕ + AvgPool (2x2)
- Flatten
- Dense (120) + ϕ
- Dense (84) + ϕ
- Dense (10)
"""

using Flux, Statistics

export FluxLeNet5, train!, predict, accuracy, evaluate

"""
    FluxLeNet5(input_dims)

LeNet-5 CNN structure with convolutional layers.

# Arguments
- `input_dims::Vector{Int}`: Input dimensions [height, width, channels]
- `ϕ`: Activation function for hidden layers

# Fields
- `input_dims::Vector{Int}`: Input dimensions specification
- `model::Flux.Chain`: The underlying Flux model
- `layer_dimensions::Vector`: Layer dimension information

Uses LeNet-5 architecture.

# Example
```julia
# Create network for 32x32 RGB images  
network = FluxLeNet5([32, 32, 3])
```
"""
mutable struct FluxLeNet5
    input_dims::Vector{Int}
    model::Flux.Chain
    layer_dimensions::Vector{Any}
    
    function FluxLeNet5(input_dims::Vector{Int}; ϕ = Flux.relu)
        height, width, channels = input_dims
        
        # Calculate layer dimensions
        layer_dimensions = Vector{Any}(undef, 9)  # Pre-allocate for 9 layers
        
        # Layer 1: Input layer
        layer_dimensions[1] = (height=height, width=width, channels=channels)
     
        # Layer 2: Convolution (5x5, 6 filters)
        layer_dimensions[2] = (height=layer_dimensions[1].height - 4, width=layer_dimensions[1].width - 4, channels=6)

        # Layer 3: Average pooling (2x2)
        layer_dimensions[3] = (height=div(layer_dimensions[2].height, 2), width=div(layer_dimensions[2].width, 2), channels=6)
        
        # Layer 4: Convolution (5x5, 16 filters)
        layer_dimensions[4] = (height=layer_dimensions[3].height - 4, width=layer_dimensions[3].width - 4, channels=16)

        # Layer 5: Average pooling (2x2)
        layer_dimensions[5] = (height=div(layer_dimensions[4].height, 2), width=div(layer_dimensions[4].width, 2), channels=16)
        
        # Layer 6: Flattened
        layer_dimensions[6] = (size=layer_dimensions[5].height * layer_dimensions[5].width * layer_dimensions[5].channels,)

        # Layer 7: Dense (120)
        layer_dimensions[7] = (size=120,)

        # Layer 8: Dense (84)
        layer_dimensions[8] = (size=84,)

        # Layer 9: Dense (10)
        layer_dimensions[9] = (size=10,)                                          
        
        model = Flux.Chain(
            # First convolutional block
            Flux.Conv((5, 5), channels => 6, ϕ),
            Flux.MeanPool((2, 2)),
            
            # Second convolutional block  
            Flux.Conv((5, 5), 6 => 16, ϕ),
            Flux.MeanPool((2, 2)),
            
            # Flatten and fully connected layers
            Flux.flatten,
            Flux.Dense(layer_dimensions[6].size, layer_dimensions[7].size, ϕ),
            Flux.Dense(layer_dimensions[7].size, layer_dimensions[8].size, ϕ),
            Flux.Dense(layer_dimensions[8].size, layer_dimensions[9].size)
        )
        
        new(input_dims, model, layer_dimensions)
    end
end

"""
    predict(network::FluxLeNet5, X)

Make predictions using trained LeNet-5 CNN.

# Arguments
- `network::FluxLeNet5`: Trained LeNet-5 network
- `X::Array{Float32, 4}`: Input data (height × width × channels × samples)

# Returns
- Predictions from the network (classes × samples)
"""
function predict(network::FluxLeNet5, X::Array{Float32, 4})
    return Flux.softmax( network.model(X) )
end

"""
    accuracy(network::FluxLeNet5, X, Y_onehot)

Calculate accuracy of LeNet-5 CNN on test data.

# Arguments
- `network::FluxLeNet5`: Trained LeNet-5 network  
- `X::Array{Float32, 4}`: Input data (height × width × channels × samples)
- `Y_onehot::Matrix{Float32}`: One-hot encoded target labels (classes × samples)

# Returns
- `Float64`: Accuracy as a fraction between 0 and 1
"""
function accuracy(network::FluxLeNet5, X::Array{Float32, 4}, Y_onehot)
    predictions = predict(network, X)
    predicted_classes = Flux.onecold(predictions) .- 1  # Convert to 0-based indexing
    true_classes = Flux.onecold(Y_onehot) .- 1  # Convert to 0-based indexing
    return mean(predicted_classes .== true_classes)
end

"""
    train!(network::FluxLeNet5, X_train, Y_train, learning_rate, epochs; batch_size=128, verbose=true)

Train a LeNet-5 CNN using mini-batch SGD.

# Arguments
- `network::FluxLeNet5`: LeNet-5 network to train
- `X_train::Array{Float32, 4}`: Training input data (height × width × channels × samples)
- `Y_train::Matrix{Float32}`: Training target labels (one-hot encoded, classes × samples)
- `learning_rate::Float64`: Adam optimizer learning rate
- `epochs::Int`: Number of training epochs
- `batch_size::Int`: Mini-batch size (default: 128)
- `verbose::Bool`: Print training progress (default: true)

# Returns
- `Vector{Float64}`: Training losses per epoch
"""
function train!(network::FluxLeNet5, X_train::Array{Float32, 4}, Y_train, learning_rate, epochs; 
               batch_size=128, verbose=true)
    
    # Define loss function and optimizer
    loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)
    optimizer = Flux.setup(Flux.Adam(learning_rate), network.model)
    
    # Create data loader for mini-batch training
    minibatches = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)
    
    losses = Float64[]
    
    for epoch in 1:epochs
        epoch_losses = Float64[]
        
        # Train on mini-batches (which are implicitly re-shuffled)
        for (x_batch, y_batch) in minibatches
            # Calculate loss and gradients for this batch
            batch_loss = loss(network.model, x_batch, y_batch)
            push!(epoch_losses, batch_loss)
            
            # Training step
            grads = Flux.gradient(m -> loss(m, x_batch, y_batch), network.model)[1]
            Flux.update!(optimizer, network.model, grads)
        end
        
        # Average loss for this epoch
        epoch_loss = mean(epoch_losses)
        push!(losses, epoch_loss)
        
        if verbose
            println("Epoch $epoch/$epochs, Loss: $(round(epoch_loss, digits=6))")
        end
    end
    
    return losses
end

"""
    evaluate(network::FluxLeNet5, X_test, Y_test, num_classes)

Comprehensive evaluation of LeNet-5 CNN with confusion matrix and per-class metrics.

# Arguments
- `network::FluxLeNet5`: Trained LeNet-5 network
- `X_test::Array{Float32, 4}`: Test input data (height × width × channels × samples)
- `Y_test::Matrix{Float32}`: Test target labels (one-hot encoded, classes × samples)  
- `num_classes::Int`: Number of classes

# Returns
- `NamedTuple`: (accuracy=Float64, predictions=Vector, true_labels=Vector, confusion_matrix=Matrix{Int})
"""
function evaluate(network::FluxLeNet5, X_test::Array{Float32, 4}, Y_test, num_classes)
    predictions = predict(network, X_test)
    predicted_classes = Flux.onecold(predictions) .- 1  # Convert to 0-based indexing
    true_classes = Flux.onecold(Y_test) .- 1  # Convert to 0-based indexing
    
    # Calculate accuracy
    accuracy = mean(predicted_classes .== true_classes)
    
    # Calculate confusion matrix
    confusion_matrix = zeros(Int, num_classes, num_classes)
    for (pred, true_class) in zip(predicted_classes, true_classes)
        confusion_matrix[true_class + 1, pred + 1] += 1  # Convert back to 1-based for indexing
    end
    
    return (
        accuracy = accuracy,
        predictions = predicted_classes,
        true_labels = true_classes,
        confusion_matrix = confusion_matrix
    )
end

