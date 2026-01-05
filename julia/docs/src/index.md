# MachineLearningCourse.jl

A Julia package for machine learning course materials and implementations.

## Installation

The recommended way to use this package is to load the module as follows:

```julia
using Pkg
Pkg.add(url="https://github.com/rajgoel/course-machine-learning", subdir="julia")
```

## Usage

To use the module type:
```julia
using MachineLearningCourse
```

To run the demos type:
```julia
using MachineLearningCourse

# Run Lecture 01 simple prediction demo
Lecture01.demo()

# Run Lecture 02 gradient descent demo
Lecture02.demo()

# etc.
```

## Course material

```@docs
MachineLearningCourse.MachineLearningCourse
```

