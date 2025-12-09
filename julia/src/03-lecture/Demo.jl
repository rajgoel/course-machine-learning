using Random

"""
    demo(seed=42, hidden_layers=[128, 64], train_size=5000, test_size=1000, learning_rate=0.001, epochs=50, verbose=true)

Vanilla MNIST handwritten digit recognition demonstration.

# Parameters
- `seed`: Random seed (default: 42)
- `hidden_layers`: Dimensions of hidden layers (default: [128, 64])
- `train_size`: Number of training samples to use (default: 5000)
- `test_size`: Number of test samples to use (default: 1000)
- `learning_rate`: Learning rate for training (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `verbose`: Print training progress (default: true)
"""
function demo(; learning_rate = 0.001, epochs = 100, seed = 42, train_size = 5000, test_size = 1000, verbose = true, hidden_layers = [128, 64])
    println("="^80)
    println("MNIST DIGIT RECOGNITION WITH DEEP NEURAL NETWORK")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Load and preprocess data
    println("\n1. Loading MNIST data...")
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = load_mnist_data(train_size, test_size)
    
    # Convert from 3D array format to vector format for vanilla DNN
    # X_train_raw is 28×28×samples, need to flatten each image and convert to vectors
    X_train = [vec(X_train_raw[:, :, i]) for i in 1:size(X_train_raw, 3)]
    Y_train = [Y_train_raw[:, i] for i in 1:size(Y_train_raw, 2)]
    X_test = [vec(X_test_raw[:, :, i]) for i in 1:size(X_test_raw, 3)]
    Y_test = [Y_test_raw[:, i] for i in 1:size(Y_test_raw, 2)]
    
    # Create network architecture for MNIST
    println("\n2. Creating network architecture...")
    # Architecture: 784 → hidden_layers → 10
    layers = [784, hidden_layers..., 10]
    dnn = DNN(layers)
    
    println("   Network architecture: $(layers)")
    total_params = sum(size(W, 1) * size(W, 2) + length(b) for (W, b) in zip(dnn.W, dnn.b))
    println("   Total parameters: $(total_params)")
    for l in 1:length(layers)-1
        layer_params = layers[l] * layers[l+1] + layers[l+1]
        println("   - Layer $l: $(layers[l]) → $(layers[l+1]) ($layer_params parameters)")
    end
    
    # Train the network
    println("\n3. Training network...")
    println("   Learning rate: $(learning_rate), Epochs: $(epochs)")
    
    losses = train!(dnn, X_train, Y_train, learning_rate, epochs, verbose)
    
    # Evaluate on test set
    println("\n4. Evaluating on test set...")
    # Get comprehensive evaluation results for MNIST (10 classes)
    results = evaluate(dnn, X_test, Y_test, 10)
    
    println("Test Accuracy: $(round(results.accuracy*100, digits=2))%")
    total = length(results.predictions)
    correct = sum(results.predictions .== results.true_labels)
    println("Correct predictions: $correct/$total")
    
    # Show confusion matrix summary
    println("\nPer-digit accuracy:")
    for digit in 1:10  # Classes 1-10 (MNIST digits 0-9)
        digit_indices = findall(x -> x == digit, results.true_labels)
        if length(digit_indices) > 0
            digit_correct = sum(results.predictions[digit_indices] .== digit)
            digit_accuracy = digit_correct / length(digit_indices)
            println("  Digit $(digit-1): $(round(digit_accuracy*100, digits=1))% ($(digit_correct)/$(length(digit_indices)))")
        end
    end
    
    # Show some example predictions
    println("\nSample predictions:")
    sample_indices = rand(1:total, min(10, total))
    for i in sample_indices
        â = predict(dnn, X_test[i])
        predicted_class = results.predictions[i] - 1  # Convert back to 0-indexed for display
        true_class = results.true_labels[i] - 1
        activation_string = join(["$d: $(round(â[d+1], digits=4))" for d in 0:9], ", ")
        status = predicted_class == true_class ? "✓" : "✗"
        println("  $status Predicted: $predicted_class, True: $true_class, Output: [ $(activation_string) ]")
    end
    
    test_accuracy = results.accuracy
    
    # Summary
    println("\n" * "="^80)
    println("TRAINING SUMMARY")
    println("="^80)
    println("Final training loss: $(round(losses[end], digits=4))")
        
    println("\nNetwork learned to classify handwritten digits! ✓")
end
