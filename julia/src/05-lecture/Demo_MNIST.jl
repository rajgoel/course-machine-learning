"""
MNIST LeNet-5 CNN Demonstration

This script demonstrates training LeNet-5 CNN on the MNIST dataset,
showing how the architecture works with grayscale handwritten digit images.
"""

using Random
using Flux
using Statistics


"""
    demo_lenet5_mnist(; seed=42, train_size=5000, test_size=1000, learning_rate=0.001, epochs=50, verbose=true)

Train and evaluate LeNet-5 CNN on MNIST dataset with grayscale images.

# Parameters
- `seed`: Random seed for reproducibility (default: 42)
- `train_size`: Number of training samples (default: 5000)
- `test_size`: Number of test samples (default: 1000)
- `learning_rate`: Learning rate for Adam optimizer (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Mini-batch size for SGD training (default: 128)
- `verbose`: Print training progress (default: true)

# Returns
- `Tuple`: (model, losses)
  - `model`: Trained LeNet-5 model for grayscale images
  - `losses`: Vector of loss values during training

This demonstration shows LeNet-5 with grayscale handwritten digit images:
- Input dimensions: 28×28×1 (grayscale) 
- First conv layer processes 1 input channel
- Dataset: MNIST handwritten digits
"""
function demo_lenet5_mnist(; seed=42, train_size=5000, test_size=1000, 
                          learning_rate=0.001, epochs=50, batch_size=128, verbose=true)
    
    println("="^80)
    println("LENET-5 CNN TRAINING ON MNIST DATASET (GRAYSCALE IMAGES)")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Load and preprocess MNIST data
    println("\n1. Loading MNIST data for CNN...")
    X_train_raw, Y_train, X_test_raw, Y_test = load_mnist_data(train_size, test_size)
    
    # Flux CNN layers require 4D tensors: (height, width, channels, batch)
    X_train = reshape(X_train_raw, 28, 28, 1, train_size)
    X_test = reshape(X_test_raw, 28, 28, 1, test_size)
    
    println("MNIST data loaded:")
    println("  Training: $(size(X_train)) (height×width×channels×samples)")
    println("  Test: $(size(X_test)) (height×width×channels×samples)")
    println("  Grayscale channels: 1 (single intensity value)")
    
    # Create LeNet-5 architecture for grayscale images
    println("\n2. Creating LeNet-5 architecture for grayscale images...")
    network = FluxLeNet5([28, 28, 1])  # 28×28×1 for MNIST
    
    println("   LeNet-5 Architecture for Grayscale Images:")
    println("   - Input: $(network.layer_dimensions[1]) (grayscale channels)")
    println("   - Conv2D: $(network.layer_dimensions[1]) → $(network.layer_dimensions[2]) (5×5 filters, ReLU)")
    println("   - AvgPool: $(network.layer_dimensions[2]) → $(network.layer_dimensions[3]) (2×2)")
    println("   - Conv2D: $(network.layer_dimensions[3]) → $(network.layer_dimensions[4]) (5×5 filters, ReLU)")
    println("   - AvgPool: $(network.layer_dimensions[4]) → $(network.layer_dimensions[5]) (2×2)")
    println("   - Flatten: $(network.layer_dimensions[5]) → $(network.layer_dimensions[6])")
    println("   - Dense: $(network.layer_dimensions[6]) → $(network.layer_dimensions[7]) (ReLU)")
    println("   - Dense: $(network.layer_dimensions[7]) → $(network.layer_dimensions[8]) (ReLU)")
    println("   - Dense: $(network.layer_dimensions[8]) → $(network.layer_dimensions[9]) (Softmax)")
    println("   Total parameters: ", sum(length, Flux.trainables(network.model)))
    
    # Train the network
    println("\n3. Training LeNet-5 on MNIST...")
    losses = train!(network, X_train, Y_train, learning_rate, epochs; 
                    batch_size=batch_size, verbose=verbose)
    
    # Evaluate on test set
    println("\n4. Evaluating LeNet-5 on MNIST test set...")
    results = evaluate(network, X_test, Y_test, 10)
    
    println("Test Accuracy: $(round(results.accuracy*100, digits=2))%")
    total = length(results.predictions)
    correct = sum(results.predictions .== results.true_labels)
    println("Correct predictions: $correct/$total")
    
    test_accuracy = results.accuracy
    
    # Summary
    println("\n" * "="^80)
    println("LENET-5 MNIST TRAINING SUMMARY")
    println("="^80)
    println("Dataset: MNIST (handwritten digits)")
    println("Input format: 28×28×1 grayscale images")
    println("Final training loss: $(round(losses[end], digits=4))")
    println("Test accuracy: $(round(test_accuracy*100, digits=2))%")
    println("Architecture: LeNet-5 adapted for grayscale input")
        
    println("\nLeNet-5 CNN on grayscale images successfully trained! ✓")
    
    return network, losses
end
