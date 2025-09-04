# MachineLearningCourse.jl

A Julia package for machine learning course materials and implementations.

## Installation

The recommended way to use this package is to clone the repository

```bash
git clone https://github.com/rajgoel/course-machine-learning
```

and load the module as follows (replace `path/to/` by the path to the repository):

```julia
using Pkg
Pkg.develop(path="path/to/course-machine-learning/julia")
```

Alternatively, you can directly load the module as follows:

```julia
using Pkg
Pkg.develop(url="https://github.com/rajgoel/course-machine-learning", subdir="julia")
```

## Usage

To use the module type:
```julia
using MachineLearningCourse
```

To access specific lecture modules (replace `XX` with the two-digit lecture number):
```julia
using MachineLearningCourse.LectureXX
```

For example, to run the demos:
```julia
using MachineLearningCourse

# Run Lecture 02 gradient descent demo
MachineLearningCourse.Lecture02.demo()

# Run Lecture 03 deep network demo
MachineLearningCourse.Lecture03.demo()
```

## Course material

### Lecture 02 - Gradient Descent

```@autodocs
Modules = [MachineLearningCourse.Lecture02]
```

### Lecture 03 - Feed Forward Networks

```@autodocs
Modules = [MachineLearningCourse.Lecture03]
```

### Lecture 04 - Deep Learning with Flux.jl

```@autodocs
Modules = [MachineLearningCourse.Lecture04]
```

