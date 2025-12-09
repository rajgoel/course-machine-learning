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
  - `X_train::Array{UInt8, 3}`: Training images (28 × 28 × samples), values 0-255
  - `Y_train::Vector{Int}`: Training labels as integers 0-9
  - `X_test::Array{UInt8, 3}`: Test images (28 × 28 × samples), values 0-255  
  - `Y_test::Vector{Int}`: Test labels as integers 0-9

# Example Usage - Image Visualization
```julia
# Load data
X_train, Y_train, X_test, Y_test = load_mnist_data(100, 50)
```
"""
function load_mnist_data(train_size::Int=5000, test_size::Int=1000)
    println("Loading MNIST dataset...")
    
    # Load MNIST data
    train_x, train_y = MNIST(split=:train)[:]  # 60,000 training samples
    test_x, test_y = MNIST(split=:test)[:]     # 10,000 test samples
    
    # Take subset but keep original format
    train_indices = randperm(size(train_x, 3))[1:train_size]
    test_indices = randperm(size(test_x, 3))[1:test_size]
    
    X_train = Float64.(train_x[:, :, train_indices]) ./ 255.0  # Normalize to [0,1]
    Y_train = reduce(hcat, one_hot_encode.(Int.(train_y[train_indices]) .+ 1, 10))

    X_test = Float64.(test_x[:, :, test_indices]) ./ 255.0  # Normalize to [0,1]
    Y_test = reduce(hcat, one_hot_encode.(Int.(test_y[test_indices]) .+ 1, 10))
    
    return X_train, Y_train, X_test, Y_test
end


