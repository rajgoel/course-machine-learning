"""
# Lecture02: Gradient Descent

A vanilla implementation of a gradient descent.

## Exported Functions

- [`demo`](@ref): 5x5 digit gardient descent demonstration

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

Run the 5x5 digit demo:

```julia
demo()
```
"""
module Lecture02

# Include the implementation files
include("GradientDescent.jl")
include("Demo.jl")

# Essential public API:
export gradient_descent
# Essential public API:
export demo

end # module Lecture03
