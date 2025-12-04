"""
Flux Deep Neural Network Implementation

Reusable Flux.jl neural network utilities extracted from the MNIST demo.
Provides the Flux equivalent of Lecture03's DNN module.
"""

using Flux
using Statistics
using Random

export FluxDNN, train!, predict, accuracy, evaluate

"""
    FluxDNN(layers)

Flux Deep Neural Network structure with fully connected layers.

# Arguments
- `layers::Vector{Int}`: Number of neurons per layer [input, hidden..., output]
- `ϕ`: Activation function for hidden layers

# Fields
- `layers::Vector{Int}`: Layer architecture specification
- `model::Flux.Chain`: The underlying Flux model

Uses ReLU activation for hidden layers, softmax activation for output layer.

# Example
```julia
# Create network: 784 inputs → 128 hidden → 64 hidden → 10 outputs
network = FluxDNN([784, 128, 64, 10])
```
"""
mutable struct FluxDNN
    layers::Vector{Int}
    model::Flux.Chain
    
    function FluxDNN(layers::Vector{Int}; ϕ = Flux.relu)
        # Build the model using Flux.Chain
        model_layers = []
        for i in 1:length(layers)-2
            # Add hidden layer with given activation function
            push!(model_layers, Flux.Dense(layers[i], layers[i+1], ϕ))
        end
        # Add last layer without activation function
        push!(model_layers, Flux.Dense(layers[end-1], layers[end]))
        
        model = Flux.Chain(model_layers...)
        new(layers, model)
    end
end

"""
    train!(network::FluxDNN, X_train, Y_train, learning_rate, epochs; batch_size=128, verbose=true)

Train a FluxDNN model using supervised learning.

# Arguments
- `network::FluxDNN`: FluxDNN network to train
- `X_train::Matrix{Float32}`: Training input data (features × samples)
- `Y_train`: Training target labels (one-hot encoded)
- `learning_rate::Float64`: Adam optimizer learning rate
- `epochs::Int`: Number of training epochs
- `batch_size::Int`: Mini-batch size (default: 128)
- `verbose::Bool`: Print training progress (default: true)

# Returns
- `Vector{Float64}`: Training losses per epoch
"""
function train!(network::FluxDNN, X_train::Matrix{Float32}, Y_train, learning_rate, epochs; 
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
        
        # Print progress
        if verbose && epoch % 10 == 0
            println("   Epoch $epoch: Loss = $(round(epoch_loss, digits=6))")
        end
    end
    
    return losses
end

"""
    train!(network::FluxDNN, X_train, Y_train, X_validate, Y_validate, learning_rate, epochs; batch_size=128, verbose=true, early_stopping_patience=10)

Train a FluxDNN model using supervised learning with validation data and early stopping.

# Arguments
- `network::FluxDNN`: FluxDNN network to train
- `X_train::Matrix{Float32}`: Training input data (features × samples)
- `Y_train`: Training target labels (one-hot encoded)
- `X_validate::Matrix{Float32}`: Validation input data (features × samples)
- `Y_validate`: Validation target labels (one-hot encoded)
- `learning_rate::Float64`: Adam optimizer learning rate
- `epochs::Int`: Number of training epochs
- `batch_size::Int`: Mini-batch size (default: 128)
- `verbose::Bool`: Print training progress (default: true)
- `early_stopping_patience::Int`: Number of epochs to wait before early stopping (default: 10)

# Returns
- `NamedTuple`: (train_losses=Vector{Float64}, val_losses=Vector{Float64})
"""
function train!(network::FluxDNN, X_train::Matrix{Float32}, Y_train, X_validate::Matrix{Float32}, Y_validate,
               learning_rate, epochs; batch_size=128, verbose=true, early_stopping_patience=10)
    
    # Define loss function and optimizer
    loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)
    optimizer = Flux.setup(Flux.Adam(learning_rate), network.model)
    
    # Create data loader for mini-batch training
    minibatches = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)
    
    train_losses = Float64[]
    val_losses = Float64[]
    best_val_loss = Inf
    patience_counter = 0
    
    for epoch in 1:epochs
        epoch_losses = Float64[]
        
        # Train on mini-batches
        for (x_batch, y_batch) in minibatches
            batch_loss = loss(network.model, x_batch, y_batch)
            push!(epoch_losses, batch_loss)
            
            grads = Flux.gradient(m -> loss(m, x_batch, y_batch), network.model)[1]
            Flux.update!(optimizer, network.model, grads)
        end
        
        # Calculate training loss for this epoch
        epoch_loss = mean(epoch_losses)
        push!(train_losses, epoch_loss)
        
        # Calculate validation loss
        val_loss = loss(network.model, X_validate, Y_validate)
        push!(val_losses, val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        if patience_counter >= early_stopping_patience
            if verbose
                println("Early stopping at epoch $epoch")
            end
            break
        end
        
        # Print progress
        if verbose && epoch % 10 == 0
            println("   Epoch $epoch: Train Loss = $(round(epoch_loss, digits=6)), Val Loss = $(round(val_loss, digits=6))")
        end
    end
    
    return (train_losses=train_losses, val_losses=val_losses)
end

"""
    predict(network::FluxDNN, X)

Make predictions using trained FluxDNN network.

# Arguments
- `network::FluxDNN`: Trained FluxDNN network
- `X::Matrix{Float32}`: Input data (features × samples)

# Returns
- Predictions from the network (output × samples)
"""
function predict(network::FluxDNN, X::Matrix{Float32})
    return Flux.softmax( network.model(X) )
end

"""
    accuracy(network::FluxDNN, X_test, Y_test)

Calculate accuracy of FluxDNN model on test data.

# Arguments
- `network::FluxDNN`: Trained FluxDNN network
- `X_test::Matrix{Float32}`: Test input data (features × samples)
- `Y_test`: Test target labels (one-hot encoded, classes × samples)

# Returns
- `Float64`: Test accuracy (0.0 to 1.0)
"""
function accuracy(network::FluxDNN, X_test::Matrix{Float32}, Y_test)
    # Get predictions
    ŷ = predict(network, X_test)
    
    # Determine class indices based on Y_test structure
    num_classes = size(Y_test, 1)
    class_indices = num_classes == 10 ? (0:9) : (1:num_classes)
    
    # Calculate overall accuracy
    return mean(Flux.onecold(ŷ, class_indices) .== Flux.onecold(Y_test, class_indices))
end

"""
    evaluate(network::FluxDNN, X_test, Y_test, classes)

Comprehensive evaluation of FluxDNN model with confusion matrix and per-class metrics.

# Arguments
- `network::FluxDNN`: Trained FluxDNN network
- `X_test::Matrix{Float32}`: Test input data (features × samples)
- `Y_test`: Test target labels (one-hot encoded, classes × samples)
- `classes`: Vector of class labels or number of classes

# Returns
- `NamedTuple`: (accuracy=Float64, predictions=Vector, true_labels=Vector, confusion_matrix=Matrix{Int})
"""
function evaluate(network::FluxDNN, X_test::Matrix{Float32}, Y_test, classes)
    # Get predictions
    ŷ = predict(network, X_test)
    
    # Determine class indices based on Y_test structure and classes argument
    num_classes = isa(classes, Integer) ? classes : length(classes)
    class_indices = (size(Y_test, 1) == 10 && num_classes == 10) ? (0:9) : (1:num_classes)
    
    # Convert to class predictions
    predictions = Flux.onecold(ŷ, class_indices)
    true_labels = Flux.onecold(Y_test, class_indices)
    
    # Calculate accuracy
    accuracy = mean(predictions .== true_labels)
    
    # Create confusion matrix
    confusion_matrix = zeros(Int, num_classes, num_classes)
    for i in 1:length(predictions)
        # Convert to 1-indexed for matrix
        true_idx = class_indices[1] == 0 ? true_labels[i] + 1 : true_labels[i]
        pred_idx = class_indices[1] == 0 ? predictions[i] + 1 : predictions[i]
        confusion_matrix[true_idx, pred_idx] += 1
    end
    
    return (accuracy=accuracy, predictions=predictions, true_labels=true_labels, confusion_matrix=confusion_matrix)
end
