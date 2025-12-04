"""
    Lecture03

Vanilla deep neural network implementation.

# Available Functions

- `demo()`: Deep learning demo for MNIST handwritten digit recognition

# Usage

```julia
using MachineLearningCourse
Lecture03.demo()
```
"""
module Lecture03

# Include the implementation files
include("DNN.jl")
include("Demo.jl")

export demo

end # module Lecture03
