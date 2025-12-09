"""
    Lecture02

Vanilla implementation of gradient descent.

# Available Functions

- `demo()`: Gradient descent demo for 5x5 pixel symbol recognition

# Usage

```julia
using MachineLearningCourse
Lecture02.demo()
```
"""
module Lecture02

# Include the implementation files
include("OneHotEncode.jl")
include("GradientDescent.jl")
include("Demo.jl")

export demo

end # module Lecture02
