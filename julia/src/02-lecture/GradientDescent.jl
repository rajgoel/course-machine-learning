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
- `x::Vector{Float64}`: Input vector for sample
- `y::Vector{Float64}`: Target output vector for sample

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: (∇W, ∇b)
  - `∇W`: Weight gradients (same size as W)
  - `∇b`: Bias gradients (same size as b)
"""
function compute_gradients(W::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64})
    # Initialize
        ∇W = zeros(Float64, size(W))
        ∇b = zeros(Float64, size(b))
  
    # Input activation 
    a = x
  
    # Compute predictions
    â = W * a + b

    # Compute ∂ℒ/∂â
    δ = ∂ℒ_∂â(y, â)

    # Compute gradients using chain rule
    # ∂ℒ/∂W = δ * a^T (outer product)
        ∇W = δ * a'
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
    gradient_descent!(W, b, X, Y)

Optimize neural network parameters using batch gradient descent.

Implements gradient descent with:
- Maximum of 10,000 iterations 
- Compute average gradients (∇W,∇b) across all samples
- Stops when ‖(∇W,∇b)‖ < tolerance 1.0e-3
- Update parameters: W ← W - η * ∇W, b ← b - η * ∇b with learning rate η = 0.1

For each iteration:
1. Compute average gradients across all training samples
2. Calculate gradient norm for convergence checking
3. Print progress information
4. Check convergence criterion
5. Update parameters using gradient descent rule

# Arguments
- `W::Matrix{Float64}`: Weight matrix (n_outputs × n_inputs), modified in-place
- `b::Vector{Float64}`: Bias vector (n_outputs,), modified in-place
- `X::Vector{Vector{Float64}}`: Training input data
- `Y::Vector{Vector{Float64}}`: Training target data (one-hot encoded)

"""
function gradient_descent!(W::Matrix{Float64}, b::Vector{Float64}, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}})
    tolerance::Float64 = 1.0e-3
    max_iterations::Int = 10000
    η::Float64 = 0.1  # learning rate  
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
        # Parameter updates: W ← W - η * ∇W, b ← b - η * ∇b
        W .-= η .* ∇W
        b .-= η .* ∇b
    end
end
