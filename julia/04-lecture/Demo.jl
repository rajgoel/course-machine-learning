"""
MNIST Digit Classification using Flux.jl Deep Neural Network

This example shows how to:
1. Load MNIST data using MLDatasets.jl
2. Build neural networks with Flux.jl
3. Train using Adam optimizer and softmax output
4. Evaluate model performance
"""

using MLDatasets
using Random
using Flux
using Statistics

include("mnist_data.jl")

"""
    evaluate_model(model, X_test, Y_test)

Evaluate Flux model performance on test data and display detailed results.

# Arguments
- `model`: Trained Flux neural network
- `X_test::Matrix{Float32}`: Test input data (features × samples)
- `Y_test`: Test target labels (one-hot encoded, classes × samples)

# Returns
- `Float64`: Test accuracy (0.0 to 1.0)

Prints comprehensive evaluation including per-digit accuracy and sample predictions.
"""
function evaluate_model(model, X_test::Matrix{Float32}, Y_test)
    # Get predictions
    ŷ = model(X_test)
    
    # Calculate overall accuracy
    accuracy = mean(Flux.onecold(ŷ, 0:9) .== Flux.onecold(Y_test, 0:9))
    
    println("Test Accuracy: $(round(accuracy*100, digits=2))%")
    
    # Convert to class predictions for detailed analysis
    predictions = Flux.onecold(ŷ, 0:9)  # Convert to class labels 0-9
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
    
    # Show some example predictions
    println("\nSample predictions:")
    sample_indices = rand(1:total, min(10, total))
    for i in sample_indices
        predicted_class = predictions[i]
        true_class = true_labels[i]
        # Get softmax probabilities for this sample
        probs = Flux.softmax(ŷ[:, i])
        activation_string = join(["$d: $(round(probs[d+1], digits=4))" for d in 0:9], ", ")
        status = predicted_class == true_class ? "✓" : "✗"
        println("  $status Predicted: $predicted_class, True: $true_class, Output: [ $(activation_string) ]")
    end
    
    return accuracy
end


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
- `verbose`: Print training progress (default: true)
"""
function demo(; seed=42, hidden_layers=[128, 64], train_size=5000, test_size=1000, 
              learning_rate=0.001, epochs=50, verbose=true)
    
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
    
    # Build the model using Flux.Chain
    model_layers = []
    for i in 1:length(layers)-2
        push!(model_layers, Flux.Dense(layers[i], layers[i+1], Flux.relu))
    end
    # Output layer with softmax
    push!(model_layers, Flux.Dense(layers[end-1], layers[end]))
    push!(model_layers, Flux.softmax)
    
    model = Flux.Chain(model_layers...)
    
    println("   Network architecture: $(layers)")
    println("   Total parameters: ", sum(length,Flux.trainables(model))  )
    for i in 1:length(layers)-1
        layer_params = layers[i] * layers[i+1] + layers[i+1]
        println("   - Layer $i: $(layers[i]) → $(layers[i+1]) ($layer_params parameters)")
    end
    
    # Define loss function and optimizer
    loss(m, x, y) = Flux.Losses.crossentropy(m(x), y)
    optimizer = Flux.setup(Flux.Adam(learning_rate), model)
    
    # Train the network
    println("\n3. Training network...")
    println("   Learning rate: $(learning_rate), Epochs: $(epochs)")
    println("   Optimizer: Adam")
    
    losses = Float64[]
    
    for epoch in 1:epochs
        # Calculate loss for this epoch
        epoch_loss = loss(model, X_train, Y_train)
        push!(losses, epoch_loss)
        
        # Training step
        grads = Flux.gradient(m -> loss(m, X_train, Y_train), model)[1]
        Flux.update!(optimizer, model, grads)
        
        # Print progress
        if verbose && epoch % 10 == 0
            println("   Epoch $epoch: Loss = $(round(epoch_loss, digits=6))")
        end
    end
    
    # Evaluate on test set
    println("\n4. Evaluating on test set...")
    test_accuracy = evaluate_model(model, X_test, Y_test)
    
    # Summary
    println("\n" * "="^80)
    println("TRAINING SUMMARY")
    println("="^80)
    println("Final training loss: $(round(losses[end], digits=4))")
    println("Test accuracy: $(round(test_accuracy*100, digits=2))%")
        
    println("\nNetwork learned to classify handwritten digits with Flux.jl! ✓")
    
    return model, losses
end
