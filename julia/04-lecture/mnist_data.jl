"""
MNIST Dataset Loading and Preprocessing

This module provides common MNIST dataset loading functionality
that can be shared across different lectures and architectures.
"""

using MLDatasets
using Random
using Flux
using Statistics

"""
    load_mnist_data(train_size=5000, test_size=1000)

Load and preprocess MNIST handwritten digit dataset.

# Arguments
- `train_size::Int`: Number of training samples to use (default: 5000)  
- `test_size::Int`: Number of test samples to use (default: 1000)

# Returns
- `Tuple`: (X_train, Y_train, X_test, Y_test)
  - `X_train::Array{Float32, 3}`: Training images (28 × 28 × samples)
  - `Y_train`: Training labels (10 × samples, one-hot encoded)
  - `X_test::Array{Float32, 3}`: Test images (28 × 28 × samples)
  - `Y_test`: Test labels (10 × samples, one-hot encoded)

Images are normalized to [0,1] and labels are one-hot encoded for 10-class classification (digits 0-9).
Returns raw 2D image format - consumers can flatten or add channels as needed.
"""
function load_mnist_data(train_size::Int=5000, test_size::Int=1000)
    println("Loading MNIST dataset...")
    
    # Load MNIST data
    train_x, train_y = MNIST(split=:train)[:]  # 60,000 training samples
    test_x, test_y = MNIST(split=:test)[:]     # 10,000 test samples
    
    println("Original data shapes:")
    println("  Training: $(size(train_x)) images, $(length(train_y)) labels")
    println("  Test: $(size(test_x)) images, $(length(test_y)) labels")
    
    # Take subset for faster training
    train_indices = randperm(size(train_x, 3))[1:train_size]
    test_indices = randperm(size(test_x, 3))[1:test_size]
    
    # Preprocess training data (28×28×samples format)
    X_train = zeros(Float32, 28, 28, train_size)
    Y_train_labels = zeros(Int, train_size)
    
    for (idx, i) in enumerate(train_indices)
        # Normalize (keep 2D structure)
        X_train[:, :, idx] = Float32.(train_x[:, :, i]) ./ 255.0f0
        Y_train_labels[idx] = Int(train_y[i])
    end
    
    # Preprocess test data
    X_test = zeros(Float32, 28, 28, test_size)
    Y_test_labels = zeros(Int, test_size)
    
    for (idx, i) in enumerate(test_indices)
        X_test[:, :, idx] = Float32.(test_x[:, :, i]) ./ 255.0f0
        Y_test_labels[idx] = Int(test_y[i])
    end
    
    # One-hot encode labels (Flux format: classes × samples)
    Y_train = Flux.onehotbatch(Y_train_labels, 0:9)
    Y_test = Flux.onehotbatch(Y_test_labels, 0:9)
    
    println("Preprocessed data:")
    println("  Training: $(size(X_train)) raw images")
    println("  Test: $(size(X_test)) raw images")
    println("  Labels: $(size(Y_train)) one-hot encoded")
    
    return X_train, Y_train, X_test, Y_test
end