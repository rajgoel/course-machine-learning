"""
MNIST Dataset Loading and Preprocessing

This module provides common MNIST dataset loading functionality
that can be shared across different lectures and architectures.
"""

using MLDatasets
using Random
using Flux
using Statistics

export load_mnist_raw, load_mnist_data

"""
    load_mnist_raw(train_size=5000, test_size=1000)

Load raw MNIST handwritten digit dataset without preprocessing.
Perfect for data exploration and visualization.

# Arguments
- `train_size::Int`: Number of training samples to use (default: 5000)  
- `test_size::Int`: Number of test samples to use (default: 1000)

# Returns
- `Tuple`: (X_train, Y_train, X_test, Y_test)
  - `X_train::Array{UInt8, 3}`: Training images (28 × 28 × samples), values 0-255
  - `Y_train::Vector{Int}`: Training labels as integers 0-9
  - `X_test::Array{UInt8, 3}`: Test images (28 × 28 × samples), values 0-255  
  - `Y_test::Vector{Int}`: Test labels as integers 0-9

# Example Usage - Image Visualization
```julia
# Load raw data
X_train, Y_train, X_test, Y_test = load_mnist_raw(100, 50)

# Display grayscale images
using ImageView
show_image = i -> imshow(X_train[:, :, i] ./ 255)
show_label = i -> Y_train[i]
show_image(1)  # Show first image
show_label(1)  # Show first label
```
"""
function load_mnist_raw(train_size::Int=5000, test_size::Int=1000)
    println("Loading raw MNIST dataset (no preprocessing)...")
    
    # Load MNIST data
    train_x, train_y = MNIST(split=:train)[:]  # 60,000 training samples
    test_x, test_y = MNIST(split=:test)[:]     # 10,000 test samples
    
    println("Original data shapes:")
    println("  Training: $(size(train_x)) images, $(length(train_y)) labels")
    println("  Test: $(size(test_x)) images, $(length(test_y)) labels")
    
    # Take subset but keep original format
    train_indices = randperm(size(train_x, 3))[1:train_size]
    test_indices = randperm(size(test_x, 3))[1:test_size]
    
    X_train = train_x[:, :, train_indices]  # Keep UInt8, 0-255
    Y_train = Int.(train_y[train_indices])  # Keep original 0-9 labels
    X_test = test_x[:, :, test_indices]     # Keep UInt8, 0-255
    Y_test = Int.(test_y[test_indices])     # Keep original 0-9 labels
    
    println("Raw data loaded:")
    println("  Training: $(size(X_train)) grayscale images (28×28×$(train_size)), values 0-255")
    println("  Test: $(size(X_test)) grayscale images (28×28×$(test_size)), values 0-255")
    println("  Labels: Integer format (0-9)")
    
    return X_train, Y_train, X_test, Y_test
end

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
    println("Loading MNIST dataset for neural network training...")
    
    # Load raw data first
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = load_mnist_raw(train_size, test_size)
    
    println("Preprocessing for neural network training...")
    
    # Convert to Float32 and normalize to [0,1]
    X_train = Float32.(X_train_raw) ./ 255.0f0
    X_test = Float32.(X_test_raw) ./ 255.0f0
    
    # One-hot encode labels (Flux format: classes × samples)
    Y_train = Flux.onehotbatch(Y_train_raw, 0:9)
    Y_test = Flux.onehotbatch(Y_test_raw, 0:9)
    
    println("Preprocessed data:")
    println("  Training: $(size(X_train)) grayscale images (28×28×$(train_size))")
    println("  Test: $(size(X_test)) grayscale images (28×28×$(test_size))")
    println("  Labels: $(size(Y_train)) one-hot encoded (10×samples)")
    println("  Pixel values normalized to [0,1]")
    
    return X_train, Y_train, X_test, Y_test
end