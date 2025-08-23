function read_data(file_path)
    inputs = Vector{Vector{Float64}}()
    outputs = Vector{OffsetArray{Float64, 1}}()
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

            # Create a 0-indexed array from 0 to 9, filled with zeros
            output = OffsetArray(zeros(Float64, 10), 0:9)
            digit = parse(Int, readline(f))  # Read the digit label
            output[digit] = 1.0  # Set desired output

            push!(inputs, input)
            push!(outputs, output)
            size += 1
        end
    end
    return size, inputs, outputs
end

function demo(sample_file)
  samples, inputs, outputs = read_data(sample_file)

  # Define initial weights and biases
  initial_weights = OffsetArray(rand(Float64, 10, 25), 0:9, 1:25)
  initial_biases = OffsetArray(rand(Float64, 10), 0:9)

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

# Example usage:
sample_file = "5x5digits.txt"  
demo(sample_file)

