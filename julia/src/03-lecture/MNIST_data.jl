using MLDatasets
using Random
using ..Lecture02: one_hot_encode

export load_mnist_data

"""
    load_mnist_data(train_size=5000, test_size=1000)

Load MNIST handwritten digit dataset.

# Arguments
- `train_size::Int`: Number of training samples to use (default: 5000)  
- `test_size::Int`: Number of test samples to use (default: 1000)

# Returns
- `Tuple`: (X_train, Y_train, X_test, Y_test)
  - `X_train::Array{Float32, 3}`: Training images (28 × 28 × samples), values 0-1
  - `Y_train::Matrix{Float32}`: Training labels as one-hot encoded matrix (10 × samples)
  - `X_test::Array{Float32, 3}`: Test images (28 × 28 × samples), values 0-1  
  - `Y_test::Matrix{Float32}`: Test labels as one-hot encoded matrix (10 × samples)

# Example Usage - Image Visualization
```julia
# Load data
X_train, Y_train, X_test, Y_test = load_mnist_data(100, 50)
```
"""
function load_mnist_data(train_size::Int=5000, test_size::Int=1000)
    println("Loading MNIST dataset...")
    
    # Load MNIST data
    X_train_all, Y_train_all = MNIST(split=:train)[:]  # 60,000 training samples
    X_test_all, Y_test_all = MNIST(split=:test)[:]     # 10,000 test samples
    
    # Take subset but keep original format
    train_indices = randperm(size(X_train_all, 3))[1:train_size]
    test_indices = randperm(size(X_test_all, 3))[1:test_size]
    
    X_train = X_train_all[:, :, train_indices]  # Already normalized to [0,1]
    Y_train = Float32.(reduce(hcat, one_hot_encode.(Int.(Y_train_all[train_indices]) .+ 1, 10)))

    X_test = X_test_all[:, :, test_indices]  # Already normalized to [0,1]
    Y_test = Float32.(reduce(hcat, one_hot_encode.(Int.(Y_test_all[test_indices]) .+ 1, 10)))
    
    return X_train, Y_train, X_test, Y_test
end


