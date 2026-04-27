"""
    Lecture03

Vanilla and Flux.jl deep neural network implementation.

# Available Functions

- `demo()`: Vanilla deep learning demo for MNIST handwritten digit recognition
- `flux_demo()`: Flux.jl deep learning demo for MNIST handwritten digit recognition

# Usage

```julia
using MachineLearningCourse
Lecture03.demo()
```

```julia
using MachineLearningCourse
Lecture03.flux_demo()
```
"""
module Lecture03

# Include the implementation files
include("MNIST_data.jl")
include("VanillaDNN.jl")
include("Demo.jl")
include("FluxDemo.jl")

export load_mnist_data, display_digit, demo, flux_demo

end # module Lecture03
