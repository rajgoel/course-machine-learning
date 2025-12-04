"""
    Lecture01

Introduction to neural networks with a simple linear classifier.

# Available Functions

- `demo(a=nothing)`: Simple demo for 2x2 pixel symbol recognition

# Usage

```julia
using MachineLearningCourse
Lecture01.demo()
```
"""
module Lecture01

include("Demo.jl")

# Re-export demo function from Demo.jl
export demo

end # module Lecture01
