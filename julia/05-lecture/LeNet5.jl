"""
LeNet-5 CNN Architecture Implementation using Flux.jl

This implementation follows the original LeNet-5 architecture:
- Conv2D (6 filters, 5x5) + ReLU + AvgPool (2x2)
- Conv2D (16 filters, 5x5) + ReLU + AvgPool (2x2)
- Flatten
- Dense (120) + ReLU
- Dense (84) + ReLU
- Dense (10) + Softmax
"""

using MLDatasets
using Random
using Flux
using Statistics

include("../04-lecture/mnist_data.jl")

"""
    create_lenet5()

Create the LeNet-5 CNN architecture using Flux.jl.

# Returns
- Flux.Chain: The LeNet-5 model

The architecture consists of:
1. Convolutional layer: 6 filters of size 5×5, ReLU activation
2. Average pooling: 2×2 pool size
3. Convolutional layer: 16 filters of size 5×5, ReLU activation
4. Average pooling: 2×2 pool size
5. Flatten layer
6. Dense layer: 120 units, ReLU activation
7. Dense layer: 84 units, ReLU activation
8. Output layer: 10 units (for 10 digit classes)
"""
function create_lenet5()
    return Flux.Chain(
        # First convolutional block
        Flux.Conv((5, 5), 1 => 6, Flux.relu),      # 28×28×1 → 24×24×6
        Flux.MeanPool((2, 2)),                      # 24×24×6 → 12×12×6
        
        # Second convolutional block  
        Flux.Conv((5, 5), 6 => 16, Flux.relu),     # 12×12×6 → 8×8×16
        Flux.MeanPool((2, 2)),                      # 8×8×16 → 4×4×16
        
        # Flatten and fully connected layers
        Flux.flatten,                               # 4×4×16 → 256
        Flux.Dense(256, 120, Flux.relu),           # 256 → 120
        Flux.Dense(120, 84, Flux.relu),            # 120 → 84
        Flux.Dense(84, 10),                        # 84 → 10
        Flux.softmax                               # Apply softmax for probabilities
    )
end

"""
    train_lenet5(; seed=42, train_size=5000, test_size=1000, learning_rate=0.001, epochs=50, verbose=true)

Train LeNet-5 CNN on MNIST dataset.

# Parameters
- `seed`: Random seed for reproducibility (default: 42)
- `train_size`: Number of training samples (default: 5000)
- `test_size`: Number of test samples (default: 1000)
- `learning_rate`: Learning rate for Adam optimizer (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `verbose`: Print training progress (default: true)

# Returns
- `Tuple`: (model, losses)
  - `model`: Trained LeNet-5 model
  - `losses`: Vector of loss values during training
"""
function train_lenet5(; seed=42, train_size=5000, test_size=1000, 
                      learning_rate=0.001, epochs=50, verbose=true)
    
    println("="^80)
    println("LENET-5 CNN TRAINING ON MNIST DATASET")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Load and preprocess data for CNN
    println("\n1. Loading MNIST data for CNN...")
    X_train_raw, Y_train, X_test_raw, Y_test = load_mnist_data(train_size, test_size)
    
    # Add channel dimension for CNN (28×28×samples → 28×28×1×samples)
    X_train = reshape(X_train_raw, 28, 28, 1, train_size)
    X_test = reshape(X_test_raw, 28, 28, 1, test_size)
    
    println("Added channel dimension for CNN:")
    println("  Training: $(size(X_train)) (height×width×channels×samples)")
    println("  Test: $(size(X_test)) (height×width×channels×samples)")
    
    # Create LeNet-5 architecture
    println("\n2. Creating LeNet-5 architecture...")
    model = create_lenet5()
    
    println("   LeNet-5 Architecture:")
    println("   - Conv2D: 1→6 filters (5×5), ReLU")
    println("   - AvgPool: 2×2")
    println("   - Conv2D: 6→16 filters (5×5), ReLU") 
    println("   - AvgPool: 2×2")
    println("   - Flatten")
    println("   - Dense: 256→120, ReLU")
    println("   - Dense: 120→84, ReLU")
    println("   - Dense: 84→10, Softmax")
    println("   Total parameters: ", sum(length, Flux.trainables(model)))
    
    # Define loss function and optimizer
    loss(m, x, y) = Flux.Losses.crossentropy(m(x), y)
    optimizer = Flux.setup(Flux.Adam(learning_rate), model)
    
    # Train the network
    println("\n3. Training LeNet-5...")
    println("   Learning rate: $(learning_rate), Epochs: $(epochs)")
    println("   Optimizer: Adam")
    
    losses = Float64[]
    
    for epoch in 1:epochs
        # Calculate loss for this epoch
        epoch_loss = loss(model, X_train, Y_train)
        push!(losses, epoch_loss)
        
        # Training step
        gradients = Flux.gradient(m -> loss(m, X_train, Y_train), model)[1]
        Flux.update!(optimizer, model, gradients)
        
        # Print progress
        if verbose && epoch % 10 == 0
            println("   Epoch $epoch: Loss = $(round(epoch_loss, digits=6))")
        end
    end
    
    # Evaluate on test set
    println("\n4. Evaluating LeNet-5 on test set...")
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
    println("LENET-5 TRAINING SUMMARY")
    println("="^80)
    println("Final training loss: $(round(losses[end], digits=4))")
    println("Test accuracy: $(round(test_accuracy*100, digits=2))%")
        
    println("\nLeNet-5 CNN successfully trained! ✓")
    
    return model, losses
end

"""
    demo_lenet5(; seed=42, train_size=5000, test_size=1000, learning_rate=0.001, epochs=50, verbose=true)

LeNet-5 CNN demonstration on MNIST dataset.

# Parameters
- `seed`: Random seed for reproducibility (default: 42)
- `train_size`: Number of training samples (default: 5000)
- `test_size`: Number of test samples (default: 1000)
- `learning_rate`: Learning rate for Adam optimizer (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `verbose`: Print training progress (default: true)
"""
function demo_lenet5(; seed=42, train_size=5000, test_size=1000, 
                     learning_rate=0.001, epochs=50, verbose=true)
    model, losses = train_lenet5(seed=seed, train_size=train_size, 
                                test_size=test_size, learning_rate=learning_rate,
                                epochs=epochs, verbose=verbose)
    return model, losses
end
