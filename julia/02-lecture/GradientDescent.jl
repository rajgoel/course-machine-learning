using LinearAlgebra

"""
    ℒ(y, â)

Mean Squared Error loss function: ℒ = ‖â - y‖².

# Arguments
- `y::Vector{Float64}`: True target values
- `â::Vector{Float64}`: Computed values

# Returns
- `Float64`: MSE loss value
"""
function ℒ(y::Vector{Float64}, â::Vector{Float64})
    diff = â - y
    return dot(diff, diff)
end

"""
    ∂ℒ_∂â(y, â)

Gradient of MSE loss with respect to computed activations: ∂ℒ/∂â = 2(â - y).

# Arguments
- `y::Vector{Float64}`: True target values
- `â::Vector{Float64}`: Computed values

# Returns
- `Vector{Float64}`: Gradient vector ∂ℒ/∂â
"""
function ∂ℒ_∂â(y::Vector{Float64}, â::Vector{Float64})
    return 2.0 * (â - y)
end

"""
    compute_gradients(W, b, x, y)

Compute gradients ∂ℒ/∂W and ∂ℒ/∂b for a single sample using backpropagation.

Calculates gradients using the chain rule:
- ∂ℒ/∂W = δ * x^T where δ = ∂ℒ/∂â
- ∂ℒ/∂b = δ

# Arguments
- `W::Matrix{Float64}`: Weight matrix (n_outputs × n_inputs)
- `b::Vector{Float64}`: Bias vector (n_outputs,)
- `x::Vector{Float64}`: Input vector for single sample
- `y::Vector{Float64}`: Target output vector for single sample

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: (∇W, ∇b)
  - `∇W`: Weight gradients (same size as W)
  - `∇b`: Bias gradients (same size as b)
"""
function compute_gradients(W::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})
    # Initialize gradients
        ∇W = zeros(Float64, size(W))
        ∇b = zeros(Float64, size(b))

    # Compute predictions
    â = W * x + b

    # Compute ∂ℒ/∂â
    δ = ∂ℒ_∂â(y, â)

    # Compute gradients using chain rule
    # ∂ℒ/∂W = δ * x^T (outer product)
        ∇W = δ * x'
    # ∂ℒ/∂b = δ
        ∇b = δ

    return ∇W, ∇b
end

"""
    compute_average_gradients(W, b, X, Y)

Compute average gradients across all training samples for batch gradient descent.

Performs gradient computation for each sample and averages the results:
- For each sample (x_i, y_i): compute ∇W_i, ∇b_i
- Return average: (1/N) * Σ(∇W_i), (1/N) * Σ(∇b_i)

# Arguments
- `W::Matrix{Float64}`: Weight matrix (n_outputs × n_inputs)
- `b::Vector{Float64}`: Bias vector (n_outputs,)
- `X::Vector{Vector{Float64}}`: Training input data (N samples)
- `Y::Vector{Vector{Float64}}`: Training target data (N samples)

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: (avg_∇W, avg_∇b)
  - `avg_∇W`: Average weight gradients
  - `avg_∇b`: Average bias gradients
"""
function compute_average_gradients(W::Matrix{Float64}, b::Vector{Float64}, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}})
    # Initialize average gradients
    sum_∇W = zeros(Float64, size(W))
    sum_∇b = zeros(Float64, size(b))

    for j in 1:length(X)
                ∇W, ∇b = compute_gradients(W, b, X[j], Y[j])
        sum_∇W .+= ∇W
        sum_∇b .+= ∇b
    end
    
    return sum_∇W / length(X), sum_∇b / length(X)
end

"""
    gradient_norm(∇W, ∇b)

Compute the Euclidean norm of the combined gradient vector.

Flattens and concatenates weight and bias gradients into a single vector,
then computes ‖∇‖ = √(‖∇W‖² + ‖∇b‖²) for convergence monitoring.

# Arguments
- `∇W::Matrix{Float64}`: Weight gradients
- `∇b::Vector{Float64}`: Bias gradients

# Returns
- `Float64`: Euclidean norm of the combined gradient vector
"""
function gradient_norm(∇W, ∇b)
    # Flatten and concatenate the gradients
    flat_weights = vec(∇W)
    flat_biases = vec(∇b)
    combined_gradient = vcat(flat_weights, flat_biases)
    
    # Return the norm of the combined gradient
    return norm(combined_gradient)
end

"""
    gradient_descent(W_0, b_0, X, Y)

Optimize neural network parameters using batch gradient descent.

Implements the complete gradient descent algorithm:
1. Initialize parameters from W_0, b_0
2. For each iteration:
   - Compute average gradients across all samples
   - Update parameters: W ← W - α * ∇W, b ← b - α * ∇b
   - Check convergence using gradient norm
3. Stop when ‖∇‖ < tolerance or max iterations reached

# Arguments
- `W_0::Matrix{Float64}`: Initial weight matrix (n_outputs × n_inputs)
- `b_0::Vector{Float64}`: Initial bias vector (n_outputs,)
- `X::Vector{Vector{Float64}}`: Training input data
- `Y::Vector{Vector{Float64}}`: Training target data (one-hot encoded)

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: (W, b)
  - `W`: Optimized weight matrix
  - `b`: Optimized bias vector

# Implementation Details
- Learning rate: α = 0.1
- Convergence tolerance: 1e-3
- Maximum iterations: 10,000
- Uses MSE loss function: ℒ = ‖â - y‖²
"""
function gradient_descent(W_0::Matrix{Float64}, b_0::Vector{Float64}, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}})
    W = copy(W_0)
    b = copy(b_0)
    
    tolerance::Float64 = 1.0e-3
    max_iterations::Int = 1e4
    α::Float64 = 0.1  # learning rate
    
    for iter in 1:max_iterations
        # Compute the average gradients ∇W and ∇b
                ∇W, ∇b = compute_average_gradients(W, b, X, Y)
        grad_norm = gradient_norm(∇W, ∇b)
        
        println("Iteration $iter, ‖(∇W,∇b)‖ = $grad_norm")
        
        # Break if the gradient norm is smaller than the tolerance
        if grad_norm < tolerance
            println("Gradient norm below tolerance. Stopping.")
            break
        end
        
        # Parameter updates: W ← W - α * ∇W, b ← b - α * ∇b
        W .-= α .* ∇W
        b .-= α .* ∇b
    end
    
    return W, b
end
