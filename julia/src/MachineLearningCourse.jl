"""
MachineLearningCourse

A Julia package for machine learning course materials and implementations.

## Installation and Usage

To use this package, replace `/path/to/course-machine-learning/julia` with the actual path to the project directory and run:


```julia
using Pkg
Pkg.develop(path="/path/to/course-machine-learning/julia")
```


To run the demo from Lecture03

```julia
using MachineLearningCourse.Lecture03
demo()
```

or 

```julia
using MachineLearningCourse
MachineLearningCourse.Lecture03.demo()
```


## Available implementations

- [`Lecture03`](@ref MachineLearningCourse.Lecture03): Feed forward networks
"""
module MachineLearningCourse

# Load submodules here:
#include("../02-lecture/Lecture02.jl")
include("../03-lecture/Lecture03.jl")

#using .Lecture02
using .Lecture03

end

