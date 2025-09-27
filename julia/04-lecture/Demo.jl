"""
MNIST Digit Classification using Flux.jl Deep Neural Network

This example shows how to:
1. Load MNIST data using MLDatasets.jl
2. Build neural networks with Flux.jl
3. Train using Adam optimizer and softmax output
4. Evaluate model performance
"""

using Random
using Flux
using Statistics

include("mnist_data.jl")


"""
    demo(seed=42, hidden_layers=[128, 64], train_size=5000, test_size=1000, learning_rate=0.001, epochs=50, verbose=true)

MNIST handwritten digit recognition demonstration using Flux.jl.

# Parameters
- `seed`: Random seed (default: 42)
- `hidden_layers`: Dimensions of hidden layers (default: [128, 64])
- `train_size`: Number of training samples to use (default: 5000)
- `test_size`: Number of test samples to use (default: 1000)
- `learning_rate`: Learning rate for Adam optimizer (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Mini-batch size for SGD training (default: 128)
- `verbose`: Print training progress (default: true)
"""
function demo(; seed=42, hidden_layers=[128, 64], train_size=5000, test_size=1000, 
              learning_rate=0.001, epochs=50, batch_size=128, verbose=true)
    
    println("="^80)
    println("MNIST DIGIT RECOGNITION WITH FLUX.JL DEEP NEURAL NETWORK")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Load and preprocess data
    println("\n1. Loading MNIST data...")
    X_train_raw, Y_train, X_test_raw, Y_test = load_mnist_data(train_size, test_size)
    
    # Flatten images for fully connected network (28×28×samples → 784×samples)
    X_train = reshape(X_train_raw, 784, train_size)
    X_test = reshape(X_test_raw, 784, test_size)
    
    println("Flattened for fully connected network:")
    println("  Training: $(size(X_train)) (features × samples)")
    println("  Test: $(size(X_test)) (features × samples)")
    
    # Create network architecture for MNIST using Flux
    println("\n2. Creating Flux network architecture...")
    layers = [784, hidden_layers..., 10]
    network = FluxDNN(layers)
    
    println("   Network architecture: $(layers)")
    println("   Total parameters: ", sum(length,Flux.trainables(network.model)))
    for i in 1:length(layers)-1
        layer_params = layers[i] * layers[i+1] + layers[i+1]
        println("   - Layer $i: $(layers[i]) → $(layers[i+1]) ($layer_params parameters)")
    end
    
    # Train the network
    println("\n3. Training network with mini-batch SGD...")
    println("   Learning rate: $(learning_rate), Epochs: $(epochs)")
    println("   Batch size: $(batch_size), Optimizer: Adam")
    
    losses = train!(network, X_train, Y_train, learning_rate, epochs; 
                    batch_size=batch_size, verbose=verbose)
    
    # Evaluate on test set
    println("\n4. Evaluating on test set...")
    # Get comprehensive evaluation results for MNIST (10 classes)
    results = evaluate(network, X_test, Y_test, 10)
    
    println("Test Accuracy: $(round(results.accuracy*100, digits=2))%")
    total = length(results.predictions)
    correct = sum(results.predictions .== results.true_labels)
    println("Correct predictions: $correct/$total")
    
    # Show confusion matrix summary
    println("\nPer-digit accuracy:")
    for digit in 0:9  # MNIST digits 0-9
        digit_indices = findall(x -> x == digit, results.true_labels)
        if length(digit_indices) > 0
            digit_correct = sum(results.predictions[digit_indices] .== digit)
            digit_accuracy = digit_correct / length(digit_indices)
            println("  Digit $digit: $(round(digit_accuracy*100, digits=1))% ($(digit_correct)/$(length(digit_indices)))")
        end
    end
    
    # Show some example predictions
    println("\nSample predictions:")
    sample_indices = rand(1:total, min(10, total))
    for i in sample_indices
        ŷ = network.model(X_test[:, i:i])  # Get prediction for single sample
        predicted_class = results.predictions[i]
        true_class = results.true_labels[i]
        probs = Flux.softmax(ŷ[:, 1])
        activation_string = join(["$d: $(round(probs[d+1], digits=4))" for d in 0:9], ", ")
        status = predicted_class == true_class ? "✓" : "✗"
        println("  $status Predicted: $predicted_class, True: $true_class, Output: [ $(activation_string) ]")
    end
    
    test_accuracy = results.accuracy
    
    # Summary
    println("\n" * "="^80)
    println("TRAINING SUMMARY")
    println("="^80)
    println("Final training loss: $(round(losses[end], digits=4))")
    println("Test accuracy: $(round(test_accuracy*100, digits=2))%")
        
    println("\nNetwork learned to classify handwritten digits with Flux.jl! ✓")
    
    return network, losses
end
