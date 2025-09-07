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
    model, layer_dimensions = create_lenet5(32, 32, 3)  # 32×32×3 for SVHN
    
    println("   LeNet-5 Architecture for RGB Images:")
    println("   - Input: $(layer_dimensions[1]) (RGB channels)")
    println("   - Conv2D: $(layer_dimensions[1]) → $(layer_dimensions[2]) (5×5 filters, ReLU)")
    println("   - AvgPool: $(layer_dimensions[2]) → $(layer_dimensions[3]) (2×2)")
    println("   - Conv2D: $(layer_dimensions[3]) → $(layer_dimensions[4]) (5×5 filters, ReLU)")
    println("   - AvgPool: $(layer_dimensions[4]) → $(layer_dimensions[5]) (2×2)")
    println("   - Flatten: $(layer_dimensions[5]) → $(layer_dimensions[6])")
    println("   - Dense: $(layer_dimensions[6]) → $(layer_dimensions[7]) (ReLU)")
    println("   - Dense: $(layer_dimensions[7]) → $(layer_dimensions[8]) (ReLU)")
    println("   - Dense: $(layer_dimensions[8]) → $(layer_dimensions[9]) (Softmax)")
    println("   Total parameters: ", sum(length, Flux.trainables(model)))
    
    # Define loss function and optimizer
    loss(m, x, y) = Flux.Losses.crossentropy(m(x), y)
    optimizer = Flux.setup(Flux.Adam(learning_rate), model)
    
    # Create data loader for mini-batch training
    println("\n3. Creating mini-batch data loader...")
    train_loader = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)
    
    # Train the network
    println("\n4. Training LeNet-5 on SVHN with mini-batch SGD...")
    println("   Learning rate: $(learning_rate), Epochs: $(epochs)")
    println("   Batch size: $(batch_size), Optimizer: Adam")
    println("   Dataset: Street View House Numbers (real-world RGB images)")
    
    losses = Float64[]
    
    for epoch in 1:epochs
        epoch_losses = Float64[]
        
        # Train on mini-batches
        for (x_batch, y_batch) in train_loader
            # Calculate loss and gradients for this batch
            batch_loss = loss(model, x_batch, y_batch)
            push!(epoch_losses, batch_loss)
            
            # Training step
            gradients = Flux.gradient(m -> loss(m, x_batch, y_batch), model)[1]
            Flux.update!(optimizer, model, gradients)
        end
        
        # Average loss for this epoch
        epoch_loss = mean(epoch_losses)
        push!(losses, epoch_loss)
        
        # Print progress
        if verbose && epoch % 10 == 0
            println("   Epoch $epoch: Loss = $(round(epoch_loss, digits=6))")
        end
    end
    
    # Evaluate on test set
    println("\n5. Evaluating LeNet-5 on SVHN test set...")
    ŷ = model(X_test)
    test_accuracy = mean(Flux.onecold(ŷ, 0:9) .== Flux.onecold(Y_test, 0:9))
    
    println("Test Accuracy: $(round(test_accuracy*100, digits=2))%")
    
    # Convert to class predictions for detailed analysis
    predictions = Flux.onecold(ŷ, 0:9)
    true_labels = Flux.onecold(Y_test, 0:9)
    
    total = length(predictions)
    correct = sum(predictions .== true_labels)
    println("Correct predictions: $correct/$total")
    
    # Show per-digit accuracy
    println("\nPer-digit accuracy:")
    for digit in 0:9
        digit_indices = findall(x -> x == digit, true_labels)
        if length(digit_indices) > 0
            digit_correct = sum(predictions[digit_indices] .== digit)
            digit_accuracy = digit_correct / length(digit_indices)
            println("  Digit $digit: $(round(digit_accuracy*100, digits=1))% ($(digit_correct)/$(length(digit_indices)))")
        end
    end
    
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
    
    return model, losses
end
