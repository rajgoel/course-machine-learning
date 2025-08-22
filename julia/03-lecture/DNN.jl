"""
Deep Neural Network Implementation in Julia
"""

using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(42)

"""
Deep Neural Network structure
- layers: vector specifying number of neurons per layer [input, hidden..., output]
- W: array of weight matrices W^[l]
- b: array of bias vectors b^[l]
- L: total number of layers (input, hidden..., output)
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
Activation function σ
ReLU: σ(z) = max(0,z)
"""
function σ(z::Real)
    # ReLU activation function
    return max(0.0, z)
end

"""
∂σ/∂z derivative
ReLU: ∂σ/∂z = 1 if z > 0, 0 if z ≤ 0
"""
function ∂σ_∂z(z::Real)
    return z > 0 ? 1.0 : 0.0
end

"""
Forward propagation
Computes activations a^[l] for all layers

Mathematical formulation:
z^[l] = W^[l] * a^[l-1] + b^[l]
a^[l] = σ(z^[l]) = σ(z^[l])

Returns:
- activations: a^[0], a^[1], ..., a^[L-1]
- z_values: z^[1], z^[2], ..., z^[L-1]
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
Mean Squared Error loss function
ℒ = ||y - ŷ||²
"""
function ℒ(y_true::Vector{Float64}, y_pred::Vector{Float64})
    diff = y_pred - y_true
    return dot(diff, diff)
end

"""
∂ℒ/∂ŷ = 2(ŷ - y)
"""
function ∂ℒ_∂ŷ(y_true::Vector{Float64}, y_pred::Vector{Float64})
    return  2.0*(y_pred - y_true)
end

"""
Backpropagation algorithm
Computes gradients ∂ℒ/∂W^[l] and ∂ℒ/∂b^[l] for all layers

Gradients: ∂ℒ/∂W^[l] = δ^[l] * (a^[l-1])^T
           ∂ℒ/∂b^[l] = δ^[l]
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
Gradient descent parameter update
W^[l] ← W^[l] - α * ∂ℒ/∂W^[l]
b^[l] ← b^[l] - α * ∂ℒ/∂b^[l]
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
Training function
Implements the complete training algorithm:
1. Forward propagation
2. Loss computation  
3. Backpropagation
4. Parameter update
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
Prediction function
Performs forward propagation to get network output ŷ
"""
function predict(network::DNN, x::Vector{Float64})
    activations, _ = forwardpropagation(network, x)
    return activations[end]  # Return a^[L-1] = ŷ
end

