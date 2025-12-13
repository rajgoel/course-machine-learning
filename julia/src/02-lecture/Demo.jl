"""
    read_data(file_path)

Read 5x5 digit training data from a text file.

File format: Each digit consists of 6 lines:
- 5 lines of 5 space-separated Float32 values (5x5 pixel grid)
- 1 line with the digit label (0-9)

# Arguments
- `file_path::String`: Path to the data file

# Returns
- `Tuple{Vector{Vector{Float32}}, Vector{Vector{Float32}}}`: (X, Y)
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
    X = Vector{Vector{Float32}}()
    Y = Vector{Vector{Float32}}()  # Changed from OffsetArray to regular Vector
    size = 0
    open(file_path, "r") do f
        while !eof(f)
            # Read 5 rows of 5 values for the digit grid
            input = Vector{Float32}()
            for _ in 1:5  # Read 5 lines for a 5x5 grid
                line = readline(f)
                # Parse the line into Float32 and append to input
                append!(input, parse.(Float32, split(line)))
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
- Forward pass: â = W * a + b
- Sample loss: ℒ(y, â) = ‖â - y‖²
- Total loss: Σ ℒ(y_i, â_i) over all samples

# Arguments
- `W::Matrix{Float32}`: Weight matrix (n_outputs × n_inputs)
- `b::Vector{Float32}`: Bias vector (n_outputs,)
- `X::Vector{Vector{Float32}}`: Training input data
- `Y::Vector{Vector{Float32}}`: Training target data (one-hot encoded)

# Returns
- `Float32`: Total loss across all training samples
"""
function total_loss(W::Matrix{Float32}, b::Vector{Float32}, X::Vector{Vector{Float32}}, Y::Vector{Vector{Float32}})
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

Loads training data, randomly initializes weights and biases, and uses gradient descent to minimize total loss. Prints initial and final loss values.

# Example
```julia
demo()  # Uses sample data file
```
"""
function demo(sample_file=joinpath(@__DIR__, "5x5digits.txt"))
  X, Y = read_data(sample_file)

  # Define initial W and b
  W = rand(Float32, 10, 25)  # Regular arrays, 1-indexed
  b = rand(Float32, 10)

  initial_loss = total_loss(W, b, X, Y)

  # Perform gradient descent
  gradient_descent!(W, b, X, Y)

  println("Initial total loss: ", initial_loss)
  println("Optimized total loss: ", total_loss(W, b, X, Y))
end
