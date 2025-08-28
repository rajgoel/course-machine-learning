"""
# Lecture02: Gradient Descent

A vanilla implementation of gradient descent.

## Exported Functions

- [`demo`](@ref): 5x5 digit gradient descent demonstration

## Usage Examples

Apply gradient descent:

```julia
using MachineLearningCourse.Lecture02

demo()
```
"""
module Lecture02

# Include the implementation files
include("GradientDescent.jl")
include("Demo.jl")

# Essential public API:
export gradient_descent, one_hot_encode, ℒ, ∂ℒ_∂â
# Essential public API:
export demo

end # module Lecture03
