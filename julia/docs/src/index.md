# MachineLearningCourse.jl

A Julia package for machine learning course materials and implementations.

## Installation

To use this package from any directory:

```julia
using Pkg
Pkg.develop(url="https://github.com/rajgoel/course-machine-learning", subdir="julia")
using MachineLearningCourse
```

Or clone and develop locally:

```julia
# In terminal: git clone https://github.com/rajgoel/course-machine-learning
using Pkg
Pkg.develop(path="path/to/course-machine-learning/julia")
using MachineLearningCourse
```

## API Reference

```@autodocs
Modules = [MachineLearningCourse, MachineLearningCourse.Lecture03]
```

