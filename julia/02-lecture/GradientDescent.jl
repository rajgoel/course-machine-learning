using OffsetArrays
using LinearAlgebra

function get_sum_squared_errors(weights, biases, samples, inputs, outputs)
    sum_squared_errors = 0.0
    for j in 1:samples
        for i in 0:length(biases)-1
            sum_squared_errors += ( dot(weights[i,:], inputs[j]) + biases[i] - outputs[j][i] )^2
        end
    end
    return sum_squared_errors
end

function get_gradients(weights, biases, input, output)
    # Initialize gradients
    gradient_weights = OffsetArray(zeros(Float64, 10, 25), 0:9, 1:25)
    gradient_biases = OffsetArray(zeros(Float64, 10), 0:9)

    for i in 0:length(biases)-1
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
