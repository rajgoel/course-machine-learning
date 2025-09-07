"""
LeNet-5 CNN Architecture Implementation using Flux.jl

This implementation follows the original LeNet-5 architecture (but uses ReLU instead of sigmoid):
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
    create_lenet5(height, width, channels)

Create the LeNet-5 CNN architecture using Flux.jl. The original LeNet-5 architecture uses sigmoid, here ReLU is used.

# Parameters
- `height`: Input height in pixels
- `width`: Input width in pixels
- `channels`: The number of input channels - 1 for grayscale, 3 for RGB

# Returns
- Tuple: (model, layer_dimensions)
  - model: Flux.Chain LeNet-5 model  
  - layer_dimensions: Array of tuples showing tensor dimensions at each layer

The architecture consists of:
1. Convolutional layer: 6 filters of size 5×5, ReLU activation
2. Average pooling: 2×2 pool size
3. Convolutional layer: 16 filters of size 5×5, ReLU activation
4. Average pooling: 2×2 pool size
5. Flatten layer
6. Dense layer: calculated size → 120 units, ReLU activation
7. Dense layer: 120 → 84 units, ReLU activation
8. Output layer: 84 → 10 units (for 10 digit classes)
"""
function create_lenet5(height, width, channels)
    # Calculate layer dimensions
    layer_dimensions = Vector{Any}(undef, 9)  # Pre-allocate for 9 layers
    
    # Layer 1: Input layer
    layer_dimensions[1] = (height=height, width=width, channels=channels)
 
    # Layer 2: Convolution (5x5, 6 filters)
    layer_dimensions[2] = (height=layer_dimensions[1].height - 4, width=layer_dimensions[1].width - 4, channels=6)

    # Layer 3: Average pooling (2x2)
    layer_dimensions[3] = (height=div(layer_dimensions[2].height, 2), width=div(layer_dimensions[2].width, 2), channels=6)
    
    # Layer 4: Convolution (5x5, 16 filters)
    layer_dimensions[4] = (height=layer_dimensions[3].height - 4, width=layer_dimensions[3].width - 4, channels=16)

    # Layer 5: Average pooling (2x2)
    layer_dimensions[5] = (height=div(layer_dimensions[4].height, 2), width=div(layer_dimensions[4].width, 2), channels=16)
    
    # Layer 6: Flattened
    layer_dimensions[6] = (size=layer_dimensions[5].height * layer_dimensions[5].width * layer_dimensions[5].channels,)

    # Layer 7: Dense (120)
    layer_dimensions[7] = (size=120,)

    # Layer 8: Dense (84)
    layer_dimensions[8] = (size=84,)

    # Layer 9: Dense (10)
    layer_dimensions[9] = (size=10,)                                          
    
    model = Flux.Chain(
        # First convolutional block
        Flux.Conv((5, 5), channels => 6, Flux.relu),
        Flux.MeanPool((2, 2)),
        
        # Second convolutional block  
        Flux.Conv((5, 5), 6 => 16, Flux.relu),
        Flux.MeanPool((2, 2)),
        
        # Flatten and fully connected layers
        Flux.flatten,
        Flux.Dense(layer_dimensions[6].size, layer_dimensions[7].size, Flux.relu),
        Flux.Dense(layer_dimensions[7].size, layer_dimensions[8].size, Flux.relu),
        Flux.Dense(layer_dimensions[8].size, layer_dimensions[9].size),
        Flux.softmax
    )
    
    return model, layer_dimensions
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
    
    # Flux CNN layers require 4D tensors: (height, width, channels, batch)
    X_train = reshape(X_train_raw, 28, 28, 1, train_size)
    X_test = reshape(X_test_raw, 28, 28, 1, test_size)
    
    println("Added channel dimension for CNN:")
    println("  Training: $(size(X_train)) (height×width×channels×samples)")
    println("  Test: $(size(X_test)) (height×width×channels×samples)")
    
    # Create LeNet-5 architecture
    println("\n2. Creating LeNet-5 architecture...")
    model, layer_dimensions = create_lenet5(28, 28, 1)
    
    println("   LeNet-5 Architecture with Layer Dimensions:")
    println("   - Input: $(layer_dimensions[1])")
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
function demo_lenet5(; seed=42, train_size=5000, test_size=1000, learning_rate=0.001, epochs=50, verbose=true)
    model, losses = train_lenet5(; seed, train_size, test_size, learning_rate, epochs, verbose)
    return model, losses
end
