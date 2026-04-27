using Random
using Flux
using Statistics
using Plots
using ..Lecture03: load_mnist_data

"""
    demo(seed=42, hidden_layers=[128, 64], train_size=5000, test_size=1000, η=0.001, epochs=50, verbose=true, validation_size=0, patience=10)

MNIST handwritten digit recognition demonstration using Flux.jl.

# Parameters
- `seed`: Random seed (default: 42)
- `hidden_layers`: Dimensions of hidden layers (default: [128, 64])
- `train_size`: Number of training samples to use (default: 5000)
- `test_size`: Number of test samples to use (default: 1000)
- `η`: Learning rate for Adam optimizer (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Mini-batch size for SGD training (default: 128)
- `verbose`: Print training progress (default: true)
- `validation_size`: Number of validation samples for early stopping (default: 0 - no validation)
- `patience`: Number of epochs to wait before early stopping (default: 10)
"""
function demo(; seed=42, hidden_layers=[128, 64], train_size=5000, test_size=1000, 
              η=0.001, epochs=50, batch_size=128, verbose=true, validation_size=0, patience=10)
    
    println("="^60)
    println("MNIST DIGIT RECOGNITION WITH FLUX.JL DEEP NEURAL NETWORK")
    println("="^60)
    
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Load and preprocess data
    println("\n1. Loading MNIST data...")
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = load_mnist_data(train_size + validation_size, test_size)
    
    # Split data if validation is requested
    if validation_size > 0
        X_train = reshape(X_train_raw[:, :, 1:train_size], 784, train_size)
        Y_train = Y_train_raw[:, 1:train_size]
        X_val = reshape(X_train_raw[:, :, (train_size+1):end], 784, validation_size)
        Y_val = Y_train_raw[:, (train_size+1):end]
        
        println("Flattened for fully connected network:")
        println("  Training: $(size(X_train)) (features × samples)")
        println("  Validation: $(size(X_val)) (features × samples)")
        println("  Test: $(size(X_test_raw)) (features × samples)")
    else
        X_train = reshape(X_train_raw, 784, train_size)
        Y_train = Y_train_raw
        
        println("Flattened for fully connected network:")
        println("  Training: $(size(X_train)) (features × samples)")
        println("  Test: $(size(X_test_raw)) (features × samples)")
    end
    
    X_test = reshape(X_test_raw, 784, test_size)
    Y_test = Y_test_raw
    
    # Create network architecture for MNIST using Flux
    println("\n2. Creating Flux network architecture...")
    layers = [784, hidden_layers..., 10]
    network = DNN(layers)
    
    println("   Network architecture: $(layers)")
    println("   Total parameters: ", sum(length,Flux.trainables(network.model)))
    for i in 1:length(layers)-1
        layer_params = layers[i] * layers[i+1] + layers[i+1]
        println("   - Layer $i: $(layers[i]) → $(layers[i+1]) ($layer_params parameters)")
    end
    
    # Train the network
    println("\n3. Training network with mini-batch SGD...")
    println("   Learning rate: $(η), Epochs: $(epochs)")
    println("   Batch size: $(batch_size), Optimizer: Adam")
    
    if validation_size > 0
        println("   Using validation set with early stopping (patience: $patience)")
        losses = train!(network, X_train, Y_train, X_val, Y_val, η, epochs; 
                        batch_size=batch_size, verbose=verbose, early_stopping_patience=patience)
    else
        losses = train!(network, X_train, Y_train, η, epochs; 
                        batch_size=batch_size, verbose=verbose)
    end
    
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
    println("\n" * "="^60)
    println("TRAINING SUMMARY")
    println("="^60)
    if validation_size > 0
        println("Final training loss: $(round(losses.train_losses[end], digits=4))")
        println("Final validation loss: $(round(losses.val_losses[end], digits=4))")
    else
        println("Final training loss: $(round(losses[end], digits=4))")
    end
    println("Test accuracy: $(round(test_accuracy*100, digits=2))%")
        
    println("\nNetwork learned to classify handwritten digits with Flux.jl! ✓")
    
    return network, losses
end

"""
    plot_losses(losses; title="Training Progress")

Create a plot of training losses. Handles both single training losses and training+validation losses.

# Arguments
- `losses`: Either Vector{Float32} (training only) or NamedTuple with train_losses and val_losses
- `title`: Plot title (default: "Training Progress")

# Returns
- Plot object showing training and validation loss curves
"""
function plot_losses(losses; title="Training Progress")
    if isa(losses, NamedTuple) && haskey(losses, :train_losses) && haskey(losses, :val_losses)
        # Training and validation losses
        p = plot(losses.train_losses, label="Training Loss", lw=2, color=:firebrick, xlabel="Epoch", ylabel="Loss", title=title)
        plot!(p, losses.val_losses, label="Validation Loss", lw=2, color=:red)
    else
        # Training losses only
        p = plot(losses, label="Training Loss", lw=2, color=:firebrick, xlabel="Epoch", ylabel="Loss", title=title)
    end
    return p
end
