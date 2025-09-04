"""
# MachineLearningCourse

A Julia package for machine learning course materials and implementations.

## Content

- [`Lecture02`](@ref MachineLearningCourse.Lecture02): Gradient descent
- [`Lecture03`](@ref MachineLearningCourse.Lecture03): Feed forward networks
- [`Lecture04`](@ref MachineLearningCourse.Lecture04): Architecture and pipeline
"""
module MachineLearningCourse

# Load submodules here:
include("../02-lecture/Lecture02.jl")
include("../03-lecture/Lecture03.jl")
include("../04-lecture/Lecture04.jl")

using .Lecture02
using .Lecture03
using .Lecture04

end

