"""
    Lecture04

Deep neural network implementation using Flux.jl.

# Available Functions

- `demo()`: Deep learning demo for MNIST handwritten digit recognition

# Usage

```julia
using MachineLearningCourse
Lecture04.demo()
```

```julia
using Plots
using MachineLearningCourse
_, losses = Lecture04.demo(validation_size=1000, learning_rate=0.005, epochs=200, patience=200)
plot = Lecture04.plot_losses(losses)
plot!(plot, size=(1200, 800))
savefig(plot, "training_progress.png")
```

"""
module Lecture04

include("FluxDNN.jl")
include("Demo.jl")

export demo, plot_losses

end
