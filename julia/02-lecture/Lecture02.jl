"""
# Lecture02: Gradient Descent

A vanilla implementation of gradient descent.

## Exported Functions

- [`gradient_descent!`](@ref): Gradient descent optimization algorithm
- [`ℒ`](@ref): Mean squared error loss function 
- [`∂ℒ_∂â`](@ref): Loss gradient computation
- [`one_hot_encode`](@ref): Convert class labels to one-hot vectors

## Usage Examples

Apply gradient descent:

```julia
using MachineLearningCourse.Lecture02

# Use individual functions
W = randn(2, 3) * 0.1
b = randn(2) * 0.1
X_train = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
Y_train = [[1.0, 0.0], [0.0, 1.0]]

# Train with gradient descent
gradient_descent!(W, b, X_train, Y_train)
```
"""
module Lecture02

# Include the implementation files
include("GradientDescent.jl")
include("Demo.jl")

# Essential public API:
export gradient_descent!, one_hot_encode, ℒ, ∂ℒ_∂â

end # module Lecture02
