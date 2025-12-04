"""
Deep Neural Network Implementation in Julia
"""

using LinearAlgebra
using OffsetArrays
using Random

# Import MSE loss functions from Lecture02
using ..Lecture02: ℒ, ∂ℒ_∂â

"""
    DNN(layers)

Deep Neural Network structure with fully connected layers.

# Arguments
- `layers::Vector{Int}`: Number of neurons per layer [input, hidden..., output]

# Fields
- `layers::OffsetVector{Int}`: Layer architecture specification
- `W::Vector{Matrix{Float64}}`: Weight matrices W^[l] for each layer
- `b::Vector{Vector{Float64}}`: Bias vectors b^[l] for each layer  
- `L::Int`: Total number of layers (excluding input layer)

Uses He initialization for weights and zero initialization for biases.
ReLU activation for hidden layers, linear activation for output layer.

# Example
```julia
# Create network: 784 inputs → 128 hidden → 64 hidden → 10 outputs
network = DNN([784, 128, 64, 10])
```
"""
mutable struct DNN
    layers::OffsetVector{Int}     # [n^0, n^1, ..., n^L]
    W::Vector{Matrix{Float64}}    # Weight matrices W^l
    b::Vector{Vector{Float64}}    # Bias vectors b^l
    L::Int                        # Number of layers excluding input layer
    
    # Convenience constructor for regular Vector
    function DNN(layers::Vector{Int})
        offset_layers = OffsetArray(layers, 0:length(layers)-1)
        DNN(offset_layers)
    end

    # Constructor for OffsetVector
    function DNN(layers::OffsetVector{Int})
        L = length(layers)-1
        
        # Initialize weight matrices W^l and bias vectors b^l
        # Note: We have L weight matrices (no weights for input layer)
        W = Matrix{Float64}[]
        b = Vector{Float64}[]
        
        for l in 1:L
            # W^[l] ∈ ℝ^{n^l × n^l-1}
            W_l = randn(layers[l], layers[l-1]) * sqrt(2.0 / layers[l-1])  # He initialization
            push!(W, W_l)
            
            # b^l ∈ ℝ^{n^l}
            b_l = zeros(layers[l])
            push!(b, b_l)
        end
        
        new(layers, W, b, L)
    end 
end

"""
    ϕ(z)

ReLU activation function: ϕ(z) = max(0, z).

# Arguments
- `z::Real`: Input value

# Returns
- `Float64`: Activated value (0.0 if z ≤ 0, z if z > 0)
"""
function ϕ(z::Real)
    # ReLU activation function
    return max(0.0, z)
end

"""
    ∂ϕ_∂z(z)

Derivative of ReLU activation function: ∂ϕ/∂z = 1 if z > 0, 0 if z ≤ 0.

# Arguments
- `z::Real`: Input value

# Returns
- `Float64`: Derivative value (1.0 if z > 0, 0.0 if z ≤ 0)
"""
function ∂ϕ_∂z(z::Real)
    return z > 0 ? 1.0 : 0.0
end

"""
    forwardpropagation(network, x)

Compute forward propagation through the neural network.

Mathematical formulation:
- z^l = W^l * a^[l-1] + b^l
- a^l = ϕ(z^l) for hidden layers, a^l = z^l for output layer

# Arguments
- `network::DNN`: Neural network structure
- `x::Vector{Float64}`: Input vector

# Returns
- `Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}`: (activations, z_values)
  - `activations`: [a^0, a^1, ..., a^L] - activations for each layer
  - `z_values`: [z^1, z^2, ..., z^L] - linear combinations for each layer
"""
function forwardpropagation(network::DNN, x::Vector{Float64})
    # Initialize storage for activations and z-values
    activations = OffsetVector(Vector{Float64}[], 0:-1)
    z_values = Vector{Vector{Float64}}()
    
    # Input layer: a^0 = x
    a = copy(x)
    push!(activations, a)
    
    # Forward through hidden layers and output layer
    for l in 1:network.L  # l = 1, 2, ..., L
        # Linear transformation: z^l = W^l * a^[l-1] + b^l
        z = network.W[l] * a + network.b[l]
        push!(z_values, z)
        
        # Activation: a^l = ϕ(z^l)
        if l == network.L  # Output layer
            a = z  # Linear output
        else  # Hidden layers
            a = ϕ.(z)
        end
        push!(activations, a)
    end

    return activations, z_values
end

# Loss functions ℒ and ∂ℒ_∂â are imported from Lecture02

"""
    backpropagation(network, activations, z_values, y)

Compute gradients using backpropagation algorithm.

# Arguments
- `network::DNN`: Neural network structure
- `activations::OffsetVector{Vector{Float64}}`: Layer activations from forward pass
- `z_values::Vector{Vector{Float64}}`: Linear combinations from forward pass
- `y::Vector{Float64}`: True target values

# Returns
- `Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}}`: (∇W, ∇b)
  - `∇W`: Weight gradients for each layer
  - `∇b`: Bias gradients for each layer
"""
function backpropagation(network::DNN, activations::OffsetVector{Vector{Float64}}, 
                        z_values::Vector{Vector{Float64}}, y::Vector{Float64})

        ∇W = Matrix{Float64}[]
        ∇b = Vector{Float64}[]

    # Output error
    â = activations[end]
    δ = ∂ℒ_∂â(y, â)

    # Gradient for output layer
    pushfirst!(∇W, δ * activations[network.L-1]')
    pushfirst!(∇b, δ ) # ∂ℒ/∂b = δ

    # Backpropagation through hidden layers
    for l in network.L-1:-1:1 
        # Compute
        # - ∂ℒ_∂a^l = W^[l+1]' ∂ℒ/∂a^[l+1] for l = L-1 
        # - ∂ℒ_∂a^l = W^[l+1]' (∂ℒ/∂a^[l+1] ⨀ ∂ϕ^l/∂z^[l+1])  for l < L-1 
        δ = network.W[l+1]' * δ

        # Compute δ = ∂ℒ/∂a^l ⨀ ∂ϕ^l/∂z^l
        δ .*= ∂ϕ_∂z.(z_values[l])

        # Gradient for layer l
        pushfirst!(∇W, δ * activations[l-1]')
        pushfirst!(∇b, δ)
    end

    return ∇W, ∇b
end

"""
    update_parameters!(network, ∇W, ∇b, α)

Update network parameters using gradient descent.

Parameter updates:
- W^l ← W^l - α * ∂ℒ/∂W^l
- b^l ← b^l - α * ∂ℒ/∂b^l

# Arguments
- `network::DNN`: Neural network (modified in-place)
- `∇W::Vector{Matrix{Float64}}`: Weight gradients
- `∇b::Vector{Vector{Float64}}`: Bias gradients  
- `α::Float64`: Learning rate
"""
function update_parameters!(network::DNN, ∇W::Vector{Matrix{Float64}}, 
                           ∇b::Vector{Vector{Float64}}, α::Float64)
    
    for l in 1:network.L  # l = 1, 2, ..., L
        # Update weights: W^l ← W^l - α * ∇W^l
        network.W[l] .-= α .* ∇W[l]
        
        # Update biases: b^l ← b^l - α * ∇b^l
        network.b[l] .-= α .* ∇b[l]
    end
end

"""
    train!(network, X, Y, α=0.01, epochs=1000, verbose=true)

Train the neural network using gradient descent.

Implements the complete training algorithm:
1. Forward propagation
2. Loss computation  
3. Backpropagation
4. Parameter update

# Arguments
- `network::DNN`: Neural network (modified in-place)
- `X::Vector{Vector{Float64}}`: Training input data
- `Y::Vector{Vector{Float64}}`: Training target data
- `α::Float64`: Learning rate (default: 0.01)
- `epochs::Int`: Number of training epochs (default: 1000)
- `verbose::Bool`: Print training progress (default: true)

# Returns
- `Vector{Float64}`: Training losses for each epoch
"""
function train!(network::DNN, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}}, 
               α::Float64=0.01, epochs::Int=1000, verbose::Bool=true)
    
    losses = Float64[]
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        
        # Training loop over all samples
        for i in 1:length(X)
            # Forward propagation
            activations, z_values = forwardpropagation(network, X[i])
            
            # Compute loss
            epoch_loss += ℒ(Y[i], activations[end])
            
            # Backpropagation
                        ∇W, ∇b = backpropagation(network, activations, z_values, Y[i])
            
            # Update parameters
            update_parameters!(network, ∇W, ∇b, α)
        end
        
        # Average loss for the epoch
        avg_loss = epoch_loss / length(X)
        push!(losses, avg_loss)
        
        # Print progress
        if verbose && epoch % 10 == 0
            println("   Epoch $epoch: Average Loss = $(round(avg_loss, digits=6))")
        end
    end
    
    return losses
end

"""
    predict(network, x)

Make predictions using the trained neural network.

Performs forward propagation to compute network output ŷ.

# Arguments
- `network::DNN`: Trained neural network
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Network predictions (output layer activations)
"""
function predict(network::DNN, x::Vector{Float64})
    activations, _ = forwardpropagation(network, x)
    return activations[end]
end

"""
    accuracy(network::DNN, X_test, Y_test)

Calculate accuracy of DNN model on test data.

# Arguments
- `network::DNN`: Trained DNN network
- `X_test::Vector{Vector{Float64}}`: Test input data
- `Y_test::Vector{Vector{Float64}}`: Test target labels (one-hot encoded)

# Returns
- `Float64`: Test accuracy (0.0 to 1.0)
"""
function accuracy(network::DNN, X_test::Vector{Vector{Float64}}, Y_test::Vector{Vector{Float64}})
    correct = 0
    total = length(X_test)
    
    for i in 1:total
        â = predict(network, X_test[i])
        predicted_class = argmax(â)
        true_class = argmax(Y_test[i])
        
        if predicted_class == true_class
            correct += 1
        end
    end
    
    return correct / total
end

"""
    evaluate(network::DNN, X_test, Y_test, classes)

Comprehensive evaluation of DNN model with confusion matrix and per-class metrics.

# Arguments
- `network::DNN`: Trained DNN network
- `X_test::Vector{Vector{Float64}}`: Test input data
- `Y_test::Vector{Vector{Float64}}`: Test target labels (one-hot encoded)
- `classes`: Vector of class labels or number of classes

# Returns
- `NamedTuple`: (accuracy=Float64, predictions=Vector{Int}, true_labels=Vector{Int}, confusion_matrix=Matrix{Int})
"""
function evaluate(network::DNN, X_test::Vector{Vector{Float64}}, Y_test::Vector{Vector{Float64}}, classes)
    total = length(X_test)
    predictions = Int[]
    true_labels = Int[]
    
    for i in 1:total
        â = predict(network, X_test[i])
        predicted_class = argmax(â)
        true_class = argmax(Y_test[i])
        
        push!(predictions, predicted_class)
        push!(true_labels, true_class)
    end
    
    # Calculate accuracy
    correct = sum(predictions .== true_labels)
    accuracy = correct / total
    
    # Create confusion matrix
    num_classes = isa(classes, Integer) ? classes : length(classes)
    confusion_matrix = zeros(Int, num_classes, num_classes)
    for i in 1:total
        confusion_matrix[true_labels[i], predictions[i]] += 1
    end
    
    return (accuracy=accuracy, predictions=predictions, true_labels=true_labels, confusion_matrix=confusion_matrix)
end

