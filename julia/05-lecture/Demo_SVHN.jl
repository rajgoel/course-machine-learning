"""
SVHN LeNet-5 CNN Demonstration

This script demonstrates training LeNet-5 CNN on the SVHN (Street View House Numbers) dataset,
showing how the architecture works with RGB color images instead of grayscale MNIST.
"""

using Random
using Flux
using Statistics


"""
    demo_lenet5_svhn(; seed=42, train_size=5000, test_size=1000, learning_rate=0.001, epochs=50, verbose=true)

Train and evaluate LeNet-5 CNN on SVHN dataset with RGB color images.

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
  - `model`: Trained LeNet-5 model for RGB images
  - `losses`: Vector of loss values during training

This demonstration shows the key differences when using LeNet-5 with RGB images:
- Input dimensions: 32×32×3 (RGB) instead of 28×28×1 (grayscale)
- First conv layer processes 3 input channels instead of 1
- Dataset: Real-world street view images vs handwritten digits
"""
function demo_lenet5_svhn(; seed=42, train_size=5000, test_size=1000, 
                          learning_rate=0.001, epochs=50, batch_size=128, verbose=true)
    
    println("="^80)
    println("LENET-5 CNN TRAINING ON SVHN DATASET (RGB IMAGES)")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Load and preprocess SVHN data
    println("\n1. Loading SVHN data for CNN...")
    X_train, Y_train, X_test, Y_test = load_svhn_data(train_size, test_size)
    
    println("SVHN data loaded:")
    println("  Training: $(size(X_train)) (height×width×channels×samples)")
    println("  Test: $(size(X_test)) (height×width×channels×samples)")
    println("  RGB channels: 3 (Red, Green, Blue)")
    
    # Create LeNet-5 architecture for RGB images
    println("\n2. Creating LeNet-5 architecture for RGB images...")
    network = FluxLeNet5([32, 32, 3])  # 32×32×3 for SVHN
    
    println("   LeNet-5 Architecture for RGB Images:")
    println("   - Input: $(network.layer_dimensions[1]) (RGB channels)")
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
    println("\n3. Training LeNet-5 on SVHN...")
    losses = train!(network, X_train, Y_train, learning_rate, epochs; 
                    batch_size=batch_size, verbose=verbose)
    
    # Evaluate on test set
    println("\n4. Evaluating LeNet-5 on SVHN test set...")
    results = evaluate(network, X_test, Y_test, 10)
    
    println("Test Accuracy: $(round(results.accuracy*100, digits=2))%")
    total = length(results.predictions)
    correct = sum(results.predictions .== results.true_labels)
    println("Correct predictions: $correct/$total")
    
    test_accuracy = results.accuracy
    
    # Summary
    println("\n" * "="^80)
    println("LENET-5 SVHN TRAINING SUMMARY")
    println("="^80)
    println("Dataset: SVHN (Street View House Numbers)")
    println("Input format: 32×32×3 RGB images")
    println("Final training loss: $(round(losses[end], digits=4))")
    println("Test accuracy: $(round(test_accuracy*100, digits=2))%")
    println("Architecture: LeNet-5 adapted for RGB input")
        
    println("\nLeNet-5 CNN on RGB images successfully trained! ✓")
    println("Note: SVHN is more challenging than MNIST due to:")
    println("  - Real-world image complexity and backgrounds")
    println("  - Variable lighting and perspective")
    println("  - RGB color information to process")
    
    return network, losses
end
