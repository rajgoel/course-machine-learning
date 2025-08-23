"""
# MachineLearningCourse

A Julia package for machine learning course materials and implementations.

## Content

- [`Lecture03`](@ref MachineLearningCourse.Lecture03): Feed forward networks
"""
module MachineLearningCourse

# Load submodules here:
include("../02-lecture/Lecture02.jl")
include("../03-lecture/Lecture03.jl")

using .Lecture02
using .Lecture03

end

