"""
# Lecture03: Feed forward networks

A vanilla implementation of a deep neural network with fully connected
layers, ReLU activation for all hidden layers, and MSE loss.

## Exported Functions

- [`DNN`](@ref): Constructor for deep neural network structure
- [`train!`](@ref): Training function using backpropagation
- [`predict`](@ref): Prediction function for inference
- [`demo`](@ref): MNIST handwritten digit recognition demonstration

## Usage Examples

Create and train a neural network:

```julia
using MachineLearningCourse.Lecture03

# Create network: 2 inputs → 4 hidden → 3 hidden → 1 output
network = DNN([2, 4, 3, 1])

# Train with learning rate 0.1 for 1000 epochs
losses = train!(network, X_train, Y_train, 0.1, 1000)

# Make predictions
prediction = predict(network, x_test)
```

Run the MNIST demo:

```julia
# Default parameters (learning rate: 0.001, epochs: 50)
demo()

# Custom parameters
demo(0.005, 500)  # learning rate: 0.005, epochs: 500
```
"""
module Lecture03

# Include the implementation files
include("DNN.jl")
include("Demo.jl")

# Essential public API:
export DNN, train!, predict
# Essential public API:
export demo

end # module Lecture03
