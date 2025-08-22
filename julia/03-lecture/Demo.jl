"""
MNIST Digit Classification using Deep Neural Network
Demonstrates use of the DNN implementation

This example shows how to:
1. Load MNIST data using MLDatasets.jl
2. Preprocess data for neural networks
3. Train and evaluate on MNIST digits

Author: Asvin Goel
"""

using MLDatasets
using Random

# DNN implementation is included by the parent Lecture03 module

"""
One-hot encoding for classification
Converts class labels to one-hot vectors
Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (for 10 classes)
"""
function one_hot_encode(label::Int, num_classes::Int)
    encoded = zeros(Float64, num_classes)
    encoded[label + 1] = 1.0  # +1 because Julia is 1-indexed, MNIST labels are 0-9
    return encoded
end

function one_hot_encode_batch(labels::Vector{Int}, num_classes::Int)
    return [one_hot_encode(label, num_classes) for label in labels]
end

"""
Evaluate model performance on test data
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
Load and preprocess MNIST data
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

# Demo execution
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
