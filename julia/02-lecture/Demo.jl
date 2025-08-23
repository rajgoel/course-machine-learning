"""
    one_hot_encode(label, num_classes)

Convert class labels to one-hot vectors for classification.

# Arguments
- `label::Int`: Class label (1-indexed)
- `num_classes::Int`: Total number of classes

# Returns
- `Vector{Float64}`: One-hot encoded vector

# Example
```julia
one_hot_encode(3, 10)  # Returns [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```
"""
function one_hot_encode(label::Int, num_classes::Int)
    encoded = zeros(Float64, num_classes)
    encoded[label] = 1.0
    return encoded
end

"""
    read_data(file_path)

Read 5x5 digit training data from a text file.

File format: Each digit consists of 6 lines:
- 5 lines of 5 space-separated Float64 values (5x5 pixel grid)
- 1 line with the digit label (0-9)

# Arguments
- `file_path::String`: Path to the data file

# Returns
- `Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}}`: (X, Y)
  - `X`: Input vectors (each vector has 25 elements from 5x5 grid)
  - `Y`: One-hot encoded target vectors (10 classes, 1-indexed)

# Example
```julia
X, Y = read_data("5x5digits.txt")
# X[1] contains 25 pixel values for first digit
# Y[1] contains one-hot vector for first digit's class
```
"""
function read_data(file_path)
    X = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()  # Changed from OffsetArray to regular Vector
    size = 0
    open(file_path, "r") do f
        while !eof(f)
            # Read 5 rows of 5 values for the digit grid
            input = Vector{Float64}()
            for _ in 1:5  # Read 5 lines for a 5x5 grid
                line = readline(f)
                # Parse the line into Float64 and append to input
                append!(input, parse.(Float64, split(line)))
            end

            # Create one-hot encoded output
            digit = parse(Int, readline(f))  # Read the digit label (0-9)
            y = one_hot_encode(digit + 1, 10)  # +1 to convert 0-9 to 1-10 indexing

            push!(X, input)
            push!(Y, y)
            size += 1
        end
    end
    return X, Y
end

"""
    total_loss(W, b, X, Y)

Compute total Mean Squared Error loss across all training samples.

For each sample, performs forward pass and computes loss:
- Forward pass: â = W * x + b
- Sample loss: ℒ(y, â) = ‖â - y‖²
- Total loss: Σ ℒ(y_i, â_i) over all samples

# Arguments
- `W::Matrix{Float64}`: Weight matrix (n_outputs × n_inputs)
- `b::Vector{Float64}`: Bias vector (n_outputs,)
- `X::Vector{Vector{Float64}}`: Training input data
- `Y::Vector{Vector{Float64}}`: Training target data (one-hot encoded)

# Returns
- `Float64`: Total loss across all training samples
"""
function total_loss(W::Matrix{Float64}, b::Vector{Float64}, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}})
    sum_ℒ = 0.0
    for j in 1:length(Y)
        # Compute predictions
        â = W * X[j] + b
        
        # Add loss for current sample
        sum_ℒ  +=  ℒ(Y[j], â)
    end
    return sum_ℒ
end

"""
    demo()

Demonstration of gradient descent on 5x5 digit recognition.

Loads training data, initializes a linear classifier, and optimizes
parameters using gradient descent. Prints initial and final loss values.

# Implementation Details
- Network: Linear classifier (25 inputs → 10 outputs)
- Loss: Mean Squared Error
- Optimization: Batch gradient descent
- Random initialization for weights and biases

# Example
```julia
demo()  # Uses sample data file
```
"""
function demo(sample_file=joinpath(@__DIR__, "5x5digits.txt"))
  X, Y = read_data(sample_file)

  # Define initial W and b
  W_0 = rand(Float64, 10, 25)  # Regular arrays, 1-indexed
  b_0 = rand(Float64, 10)

  # Perform gradient descent
  W, b = gradient_descent(W_0, b_0, X, Y)

  println("Initial total loss: ", total_loss(W_0, b_0, X, Y))
  println("Optimized total loss: ", total_loss(W, b, X, Y))
end
