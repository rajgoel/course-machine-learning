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
```

## Course Structure

- **Lecture01**: Introduction to neural networks with a simple linear classifier.
- **Lecture02**: Gradient descent 
- **Lecture03**: Feed forward networks 
- **Lecture04**: MNIST handwritten digit classifier with Flux.jl
- **Lecture05**: Convolutional neural networks
- **Lecture06**: Graph neural networks
- **Lecture07**: Autoencoders
- **Lecture08**: Deep Q networks
- **Lecture09**: Proximal policy optimisation
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

end

