"""
MNIST Digit Classification using Deep Neural Network
Demonstrates use of the DNN implementation

This example shows how to:
1. Load MNIST data using MLDatasets.jl
2. Preprocess data for neural networks
3. Train and evaluate on MNIST digits
"""

using MLDatasets
using Random

"""
    one_hot_encode(label, num_classes)

Convert class labels to one-hot vectors for classification.

# Arguments
- `label::Int`: Class label (0-indexed)
- `num_classes::Int`: Total number of classes

# Returns
- `Vector{Float64}`: One-hot encoded vector

# Example
```julia
one_hot_encode(3, 10)  # Returns [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```
"""
function one_hot_encode(label::Int, num_classes::Int)
    encoded = zeros(Float64, num_classes)
    encoded[label + 1] = 1.0  # +1 because Julia is 1-indexed, MNIST labels are 0-9
    return encoded
end

"""
    one_hot_encode_batch(labels, num_classes)

Convert a batch of class labels to one-hot vectors.

# Arguments
- `labels::Vector{Int}`: Vector of class labels (0-indexed)
- `num_classes::Int`: Total number of classes

# Returns
- `Vector{Vector{Float64}}`: Vector of one-hot encoded vectors
"""
function one_hot_encode_batch(labels::Vector{Int}, num_classes::Int)
    return [one_hot_encode(label, num_classes) for label in labels]
end

"""
    evaluate_model(network, X_test, Y_test)

Evaluate neural network performance on test data and display detailed results.

# Arguments
- `network::DNN`: Trained neural network
- `X_test::Vector{Vector{Float64}}`: Test input data
- `Y_test::Vector{Vector{Float64}}`: Test target labels (one-hot encoded)

# Returns
- `Float64`: Test accuracy (0.0 to 1.0)

Prints comprehensive evaluation including per-digit accuracy, confusion matrix summary, and sample predictions.
"""
function evaluate_model(network::DNN, X_test::Vector{Vector{Float64}}, Y_test::Vector{Vector{Float64}})
    correct = 0
    total = length(X_test)
    
    predictions = Int[]
    true_labels = Int[]
    
    println("Evaluating on $total test samples...")
    
    for i in 1:total
        output = predict(network, X_test[i])
        predicted_class = argmax(output) - 1  # Convert to 0-indexed (0-9 for MNIST)
        true_class = argmax(Y_test[i]) - 1
        
        push!(predictions, predicted_class)
        push!(true_labels, true_class)
        
        if predicted_class == true_class
            correct += 1
        end
    end
    
    accuracy = correct / total
    
    println("Test Accuracy: $(round(accuracy*100, digits=2))%")
    println("Correct predictions: $correct/$total")
    
    # Show confusion matrix summary
    println("\nPer-digit accuracy:")
    for digit in 0:9
        digit_indices = findall(x -> x == digit, true_labels)
        if length(digit_indices) > 0
            digit_correct = sum(predictions[digit_indices] .== digit)
            digit_accuracy = digit_correct / length(digit_indices)
            println("  Digit $digit: $(round(digit_accuracy*100, digits=1))% ($(digit_correct)/$(length(digit_indices)))")
        end
    end
    
    # Show some example predictions
    println("\nSample predictions:")
    sample_indices = rand(1:total, min(10, total))
    for i in sample_indices
        output = predict(network, X_test[i])
        predicted_class = argmax(output) - 1  # Convert to 0-indexed (0-9 for MNIST)
        true_class = argmax(Y_test[i]) - 1
        activations = join(["$d: $(round(output[d+1], digits=4))" for d in 0:9], ", ")
        status = predicted_class == true_class ? "✓" : "✗"
        println("  $status Predicted: $predicted_class, True: $true_class, Output: [ $(activations) ]")
    end
    
    return accuracy
end

"""
    load_mnist_data(train_size=5000, test_size=1000)

Load and preprocess MNIST handwritten digit dataset.

# Arguments
- `train_size::Int`: Number of training samples to use (default: 5000)
- `test_size::Int`: Number of test samples to use (default: 1000)

# Returns
- `Tuple`: (X_train, Y_train, X_test, Y_test)
  - `X_train::Vector{Vector{Float64}}`: Training images (flattened and normalized)
  - `Y_train::Vector{Vector{Float64}}`: Training labels (one-hot encoded)
  - `X_test::Vector{Vector{Float64}}`: Test images (flattened and normalized)
  - `Y_test::Vector{Vector{Float64}}`: Test labels (one-hot encoded)

Images are flattened from 28×28 to 784-dimensional vectors and normalized to [0,1].
Labels are one-hot encoded for 10-class classification (digits 0-9).
"""
function load_mnist_data(train_size::Int=5000, test_size::Int=1000)
    println("Loading MNIST dataset...")
    
    # Load MNIST data
    train_x, train_y = MNIST.traindata()  # 60,000 training samples
    test_x, test_y = MNIST.testdata()     # 10,000 test samples
    
    println("Original data shapes:")
    println("  Training: $(size(train_x)) images, $(length(train_y)) labels")
    println("  Test: $(size(test_x)) images, $(length(test_y)) labels")
    
    # Take subset for faster training (educational purposes)
    train_indices = randperm(size(train_x, 3))[1:train_size]
    test_indices = randperm(size(test_x, 3))[1:test_size]
    
    # Preprocess training data
    X_train = Vector{Float64}[]
    Y_train = Vector{Float64}[]
    
    for i in train_indices
        # Flatten 28×28 image to 784-dimensional vector
        image = vec(train_x[:, :, i])
        # Normalize pixel values to [0, 1]
        image = Float64.(image) ./ 255.0
        push!(X_train, image)
        
        # One-hot encode label
        label = one_hot_encode(Int(train_y[i]), 10)
        push!(Y_train, label)
    end
    
    # Preprocess test data
    X_test = Vector{Float64}[]
    Y_test = Vector{Float64}[]
    
    for i in test_indices
        image = vec(test_x[:, :, i])
        image = Float64.(image) ./ 255.0
        push!(X_test, image)
        
        label = one_hot_encode(Int(test_y[i]), 10)
        push!(Y_test, label)
    end
    
    println("Preprocessed data:")
    println("  Training: $(length(X_train)) samples, $(length(X_train[1]))-dimensional inputs")
    println("  Test: $(length(X_test)) samples, $(length(Y_test[1]))-dimensional outputs")
    
    return X_train, Y_train, X_test, Y_test
end

"""
    demo(learning_rate=0.001, epochs=50)

MNIST handwritten digit recognition demonstration.

# Parameters
- `learning_rate`: Learning rate for training (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
"""
function demo(learning_rate = 0.001, epochs = 50)
    println("="^80)
    println("MNIST DIGIT RECOGNITION WITH DEEP NEURAL NETWORK")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Load and preprocess data
    println("\n1. Loading MNIST data...")
    X_train, Y_train, X_test, Y_test = load_mnist_data(5000, 1000)  # Subset for speed
    
    # Create network architecture for MNIST
    println("\n2. Creating network architecture...")
    # Architecture: 784 → 128 → 64 → 10
    mnist_network = DNN([784, 128, 64, 10])
    
    println("   Network architecture: [784, 128, 64, 10]")
    total_params = sum(size(W, 1) * size(W, 2) + length(b) for (W, b) in zip(mnist_network.W, mnist_network.b))
    println("   Total parameters: $(total_params)")
    println("   - Layer 1: 784 → 128 ($(784*128 + 128) parameters)")  
    println("   - Layer 2: 128 → 64 ($(128*64 + 64) parameters)")
    println("   - Layer 3: 64 → 10 ($(64*10 + 10) parameters)")
    
    # Train the network
    println("\n3. Training network...")
    println("   Learning rate: $(learning_rate), Epochs: $(epochs)")
    
    losses = train!(mnist_network, X_train, Y_train, learning_rate, epochs, true)
    
    # Evaluate on test set
    println("\n4. Evaluating on test set...")
    test_accuracy = evaluate_model(mnist_network, X_test, Y_test)
    
    # Summary
    println("\n" * "="^80)
    println("TRAINING SUMMARY")
    println("="^80)
    println("Final training loss: $(round(losses[end], digits=4))")
        
    println("\nNetwork learned to classify handwritten digits! ✓")
end
