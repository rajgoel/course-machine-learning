"""
Deep Neural Network Implementation in Julia
"""

using LinearAlgebra
using Random

"""
    DNN(layers)

Deep Neural Network structure with fully connected layers.

# Arguments
- `layers::Vector{Int}`: Number of neurons per layer [input, hidden..., output]

# Fields
- `layers::Vector{Int}`: Layer architecture specification
- `W::Vector{Matrix{Float64}}`: Weight matrices W^[l] for each layer
- `b::Vector{Vector{Float64}}`: Bias vectors b^[l] for each layer  
- `L::Int`: Total number of layers

Uses He initialization for weights and zero initialization for biases.
ReLU activation for hidden layers, linear activation for output layer.

# Example
```julia
# Create network: 784 inputs → 128 hidden → 64 hidden → 10 outputs
network = DNN([784, 128, 64, 10])
```
"""
mutable struct DNN
    layers::Vector{Int}           # [n^[0], n^[1], ..., n^[L-1]]
    W::Vector{Matrix{Float64}}    # Weight matrices W^[l]
    b::Vector{Vector{Float64}}    # Bias vectors b^[l]
    L::Int                        # Number of layers
    
    # Constructor
    function DNN(layers::Vector{Int})
        L = length(layers)
        
        # Initialize weight matrices W^[l] and bias vectors b^[l]
        # Note: We have L-1 weight matrices (no weights for input layer)
        W = Matrix{Float64}[]
        b = Vector{Float64}[]
        
        for l in 2:L  # l = 1, 2, ..., L-1 in our notation (Julia is 1-indexed)
            # W^[l] ∈ ℝ^{n^[l] × n^[l-1]}
            W_l = randn(layers[l], layers[l-1]) * sqrt(2.0 / layers[l-1])  # He initialization
            push!(W, W_l)
            
            # b^[l] ∈ ℝ^{n^[l]}
            b_l = zeros(layers[l])
            push!(b, b_l)
        end
        
        new(layers, W, b, L)
    end
end

"""
    σ(z)

ReLU activation function: σ(z) = max(0, z).

# Arguments
- `z::Real`: Input value

# Returns
- `Float64`: Activated value (0.0 if z ≤ 0, z if z > 0)
"""
function σ(z::Real)
    # ReLU activation function
    return max(0.0, z)
end

"""
    ∂σ_∂z(z)

Derivative of ReLU activation function: ∂σ/∂z = 1 if z > 0, 0 if z ≤ 0.

# Arguments
- `z::Real`: Input value

# Returns
- `Float64`: Derivative value (1.0 if z > 0, 0.0 if z ≤ 0)
"""
function ∂σ_∂z(z::Real)
    return z > 0 ? 1.0 : 0.0
end

"""
    forwardpropagation(network, x)

Compute forward propagation through the neural network.

Mathematical formulation:
- z^[l] = W^[l] * a^[l-1] + b^[l]
- a^[l] = σ(z^[l]) for hidden layers, a^[l] = z^[l] for output layer

# Arguments
- `network::DNN`: Neural network structure
- `x::Vector{Float64}`: Input vector

# Returns
- `Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}`: (activations, z_values)
  - `activations`: [a^[0], a^[1], ..., a^[L-1]] - activations for each layer
  - `z_values`: [z^[1], z^[2], ..., z^[L-1]] - linear combinations for each layer
"""
function forwardpropagation(network::DNN, x::Vector{Float64})
    # Initialize storage for activations and z-values
    activations = Vector{Vector{Float64}}()
    z_values = Vector{Vector{Float64}}()
    
    # Input layer: a^[0] = x
    a = copy(x)
    push!(activations, a)
    
    # Forward through hidden layers and output layer
    for l in 1:(network.L-1)  # l = 1, 2, ..., L-1
        # Linear transformation: z^[l] = W^[l] * a^[l-1] + b^[l]
        z = network.W[l] * a + network.b[l]
        push!(z_values, z)
        
        # Activation: a^[l] = σ(z^[l])
        if l == network.L - 1  # Output layer
            a = z  # Linear output
        else  # Hidden layers
            a = σ.(z)
        end
        push!(activations, a)
    end
    
    return activations, z_values
end

"""
    ℒ(y_true, y_pred)

Mean Squared Error loss function: ℒ = ||ŷ - y||².

# Arguments
- `y_true::Vector{Float64}`: True target values
- `y_pred::Vector{Float64}`: Predicted values

# Returns
- `Float64`: MSE loss value
"""
function ℒ(y_true::Vector{Float64}, y_pred::Vector{Float64})
    diff = y_pred - y_true
    return dot(diff, diff)
end

"""
    ∂ℒ_∂ŷ(y_true, y_pred)

Gradient of MSE loss with respect to predictions: ∂ℒ/∂ŷ = 2(ŷ - y).

# Arguments
- `y_true::Vector{Float64}`: True target values
- `y_pred::Vector{Float64}`: Predicted values

# Returns
- `Vector{Float64}`: Gradient vector
"""
function ∂ℒ_∂ŷ(y_true::Vector{Float64}, y_pred::Vector{Float64})
    return  2.0*(y_pred - y_true)
end

"""
    backpropagation(network, activations, z_values, y_true)

Compute gradients using backpropagation algorithm.

Calculates ∂ℒ/∂W^[l] and ∂ℒ/∂b^[l] for all layers using:
- ∂ℒ/∂W^[l] = δ^[l] * (a^[l-1])^T  
- ∂ℒ/∂b^[l] = δ^[l]

# Arguments
- `network::DNN`: Neural network structure
- `activations::Vector{Vector{Float64}}`: Layer activations from forward pass
- `z_values::Vector{Vector{Float64}}`: Linear combinations from forward pass
- `y_true::Vector{Float64}`: True target values

# Returns
- `Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}}`: (∇W, ∇b)
  - `∇W`: Weight gradients for each layer
  - `∇b`: Bias gradients for each layer
"""
function backpropagation(network::DNN, activations::Vector{Vector{Float64}}, 
                        z_values::Vector{Vector{Float64}}, y_true::Vector{Float64})

    ∇W = Matrix{Float64}[]
    ∇b = Vector{Float64}[]

    # Output error
    y_pred = activations[end]
    δ = ∂ℒ_∂ŷ(y_true, y_pred)  # (ŷ - y)

    # Gradient for last layer
    pushfirst!(∇W, δ * activations[end-1]')
    pushfirst!(∇b, copy(δ))

    # Backpropagation through hidden layers
    for l in (network.L-2):-1:1
        δ = (network.W[l+1]' * δ) .* ∂σ_∂z.(z_values[l])

        pushfirst!(∇W, δ * activations[l]')
        pushfirst!(∇b, copy(δ))
    end

    return ∇W, ∇b
end

"""
    update_parameters!(network, ∇W, ∇b, α)

Update network parameters using gradient descent.

Parameter updates:
- W^[l] ← W^[l] - α * ∂ℒ/∂W^[l]
- b^[l] ← b^[l] - α * ∂ℒ/∂b^[l]

# Arguments
- `network::DNN`: Neural network (modified in-place)
- `∇W::Vector{Matrix{Float64}}`: Weight gradients
- `∇b::Vector{Vector{Float64}}`: Bias gradients  
- `α::Float64`: Learning rate
"""
function update_parameters!(network::DNN, ∇W::Vector{Matrix{Float64}}, 
                           ∇b::Vector{Vector{Float64}}, α::Float64)
    
    for l in 1:(network.L-1)  # l = 1, 2, ..., L-1
        # Update weights: W^[l] ← W^[l] - α * ∇W^[l]
        network.W[l] .-= α .* ∇W[l]
        
        # Update biases: b^[l] ← b^[l] - α * ∇b^[l]
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
    return activations[end]  # Return a^[L-1] = ŷ
end

