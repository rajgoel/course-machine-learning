"""
# Lecture04: Flux.jl Deep Neural Networks

Implementation using Flux.jl framework for deep neural networks.

## Exported Functions

- [`FluxDNN`](@ref): Flux-based deep neural network constructor
- [`train!`](@ref): Mini-batch training with Adam optimizer
- [`predict`](@ref): Neural network inference
- [`accuracy`](@ref): Model accuracy evaluation
- [`evaluate`](@ref): Comprehensive model evaluation with confusion matrix

## Usage Examples

Create and train a Flux neural network:

```julia
using MachineLearningCourse.Lecture04

# Create network: 784 inputs → 128 hidden → 64 hidden → 10 outputs
network = FluxDNN([784, 128, 64, 10])

# Train with Adam optimizer
losses = train!(network, X_train, Y_train, 0.001, 50; batch_size=128)

# Make predictions
predictions = predict(network, X_test)

# Evaluate performance
results = evaluate(network, X_test, Y_test, 10)
println("Accuracy: \$(results.accuracy)")
```
"""
module Lecture04

include("FluxDNN.jl")
include("Demo.jl")

# Export reusable functions from FluxDNN
export FluxDNN, train!, predict, accuracy, evaluate

end
