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

function read_data(file_path)
    inputs = Vector{Vector{Float64}}()
    outputs = Vector{Vector{Float64}}()  # Changed from OffsetArray to regular Vector
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
            output = one_hot_encode(digit + 1, 10)  # +1 to convert 0-9 to 1-10 indexing

            push!(inputs, input)
            push!(outputs, output)
            size += 1
        end
    end
    return size, inputs, outputs
end

function demo(sample_file=joinpath(@__DIR__, "5x5digits.txt"))
  samples, inputs, outputs = read_data(sample_file)

  # Define initial weights and biases
  initial_weights = rand(Float64, 10, 25)  # Regular arrays, 1-indexed
  initial_biases = rand(Float64, 10)

  # Perform gradient descent
  optimized_weights, optimized_biases = gradient_descent(initial_weights, initial_biases, samples, inputs, outputs)

#  save_parameters(parameter_file, optimized_weights, optimized_biases)
  println("Initial sum of squared errors: ", get_sum_squared_errors(initial_weights,initial_biases,samples,inputs,outputs))
#  println("Initial weights: ", initial_weights)
#  println("Initial biases: ", initial_biases)
  println("Optimized sum of squared errors: ", get_sum_squared_errors(optimized_weights, optimized_biases, samples ,inputs,outputs))
#  println("Optimized weights: ", optimized_weights)
#  println("Optimized biases: ", optimized_biases)

end
