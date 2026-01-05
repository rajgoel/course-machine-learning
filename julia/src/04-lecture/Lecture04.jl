"""
    Lecture04

Stochastic gradient descent using Flux.jl.

# Available Functions

- `demo()`: Stochastic gradient descent demo for MNIST handwritten digit recognition

# Usage

```julia
using MachineLearningCourse
Lecture04.demo()
```

```julia
using Plots
using MachineLearningCourse
_, losses = Lecture04.demo(validation_size=1000, epochs=100, patience=100)
plot = Lecture04.plot_losses(losses)
plot!(plot, size=(1200, 800))
savefig(plot, "training_progress.png")
```

"""
module Lecture04

include("DNN.jl")
include("Demo.jl")

export demo, plot_losses

end
