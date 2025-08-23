using OffsetArrays
using LinearAlgebra
using JSON

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


function get_sum_squared_errors(weights, biases, samples, inputs, outputs)
    sum_squared_errors = 0.0
    for j in 1:samples
        for i in 0:9
            sum_squared_errors += ( dot(weights[i,:], inputs[j]) + biases[i] - outputs[j][i] )^2
        end
    end
    return sum_squared_errors
end

function get_gradients(weights, biases, input, output)
    # Initialize gradients
    gradient_weights = OffsetArray(zeros(Float64, 10, 25), 0:9, 1:25)
    gradient_biases = OffsetArray(zeros(Float64, 10), 0:9)

    for i in 0:9
        error = dot(weights[i,:], input) + biases[i] - output[i]
        # Gradient w.r.t weights and biases
        gradient_weights[i, :] = 2 * error * input
        gradient_biases[i] = 2 * error
    end

    return gradient_weights, gradient_biases
end

function get_average_gradient(weights, biases, samples, inputs, outputs)
    # Initialize average gradients
    sum_gradient_weights = OffsetArray(zeros(Float64, 10, 25), 0:9, 1:25)
    sum_gradient_biases = OffsetArray(zeros(Float64, 10), 0:9)

    for j in 1:samples
        gradient_weights, gradient_biases = get_gradients(weights, biases, inputs[j], outputs[j])
        sum_gradient_weights .+= gradient_weights
        sum_gradient_biases .+= gradient_biases
    end
    
    return sum_gradient_weights / samples, sum_gradient_biases / samples
end

function get_norm(gradient_weights, gradient_biases)
    # Flatten and concatenate the gradients
    flat_weights = vec(parent(gradient_weights))  # Convert weights to a 1D array
    flat_biases = vec(parent(gradient_biases))    # Convert biases to a 1D array
    combined_gradient = vcat(flat_weights, flat_biases)  # Concatenate them
    
    # Return the norm of the combined gradient
    return norm(combined_gradient)
end

function gradient_descent(initial_weights, initial_biases, samples, inputs, outputs)
    weights = copy(initial_weights)
    biases = copy(initial_biases)
    
    tolerance = 1.0e-3
    max_iterations = 1e4
    learning_rate = .1
    
    for iter in 1:max_iterations
        # Compute the average gradient
        avg_gradient_weights, avg_gradient_biases = get_average_gradient(weights, biases, samples, inputs, outputs)
        norm_gradient = get_norm(avg_gradient_weights, avg_gradient_biases)
        
        println("Iteration $iter, Norm of gradient: $norm_gradient")
        
        # Break if the gradient norm is smaller than the tolerance
        if norm_gradient < tolerance
            println("Gradient norm below tolerance. Stopping.")
            break
        end
        
        # Update weights and biases using gradient descent
        weights .-= learning_rate .* avg_gradient_weights
        biases .-= learning_rate .* avg_gradient_biases
    end
    
    return weights, biases
end

function save_parameters(parameter_file, weights, biases)
    # Prepare the list of dictionaries
    params = []
    for i in 0:9
      push!(params, Dict(
        "bias" => biases[i],
        "weights" => weights[i, :]
      ))
    end

    # Save the list of dictionaries to a JSON file
    open(parameter_file, "w") do file
      JSON.print(file, params)
    end
end

function main(sample_file,parameter_file)
  samples, inputs, outputs = read_data(sample_file)

  # Define initial weights and biases
  initial_weights = OffsetArray(rand(Float64, 10, 25), 0:9, 1:25)
  initial_biases = OffsetArray(rand(Float64, 10), 0:9)
#  initial_weights = OffsetArray(zeros(Float64, 10, 25), 0:9, 1:25)
#  initial_biases = OffsetArray(zeros(Float64, 10), 0:9)

  # Perform gradient descent
  optimized_weights, optimized_biases = gradient_descent(initial_weights, initial_biases, samples, inputs, outputs)

  save_parameters(parameter_file, optimized_weights, optimized_biases)
  println("Initial sum of squared errors: ", get_sum_squared_errors(initial_weights,initial_biases,samples,inputs,outputs))
#  println("Initial weights: ", initial_weights)
#  println("Initial biases: ", initial_biases)
  println("Optimized sum of squared errors: ", get_sum_squared_errors(optimized_weights, optimized_biases, samples ,inputs,outputs))
#  println("Optimized weights: ", optimized_weights)
#  println("Optimized biases: ", optimized_biases)

end

# Example usage:
sample_file = "5x5digits.txt"  
parameter_file = "5x5parameters.json"
main(sample_file,parameter_file)

