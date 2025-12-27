"""
# MachineLearningCourse.jl

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/rajgoel/course-machine-learning", subdir="julia")
```

## Quick Start

```julia
using MachineLearningCourse

# Run lecture demos
Lecture01.demo()
Lecture02.demo()
Lecture03.demo()
# etc.
```

## Course Structure

- **Lecture01**: A simple linear classifier
- **Lecture02**: Vanilla implementation of gradient descent
- **Lecture03**: Vanilla and Flux.jl deep neural network implementation
- **Lecture04**: Deep neural network implementation using Flux.jl
- **Lecture05**: Implementation of filtering, pooling, and convolution
- **Lecture06**: Implementation of graph convolutional networks (GCNs) for collaborative filtering
- **Lecture07**: Implementation of autoencoders
- **Lecture08**: Implementation of Deep Q-Learning for the Breakout game
- **Lecture09**: Implementation of policy gradient methods for the Breakout game
"""
module MachineLearningCourse

# Load submodules here:
include("01-lecture/Lecture01.jl")
include("02-lecture/Lecture02.jl")
include("03-lecture/Lecture03.jl")
include("04-lecture/Lecture04.jl")
include("05-lecture/Lecture05.jl")
include("06-lecture/Lecture06.jl")
include("07-lecture/Lecture07.jl")
include("08-lecture/Lecture08.jl")
include("09-lecture/Lecture09.jl")

using .Lecture01
using .Lecture02
using .Lecture03
using .Lecture04
using .Lecture05
using .Lecture06
using .Lecture07
using .Lecture08
using .Lecture09

export Lecture01, Lecture02, Lecture03, Lecture04, Lecture05, Lecture06, Lecture07, Lecture08, Lecture09
export load_mnist_data, display_digit
export load_movielens_data

end

