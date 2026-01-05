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

## Lectures

- [Lecture 01](lecture01.md) - A simple linear classifier
- [Lecture 02](lecture02.md) - Vanilla implementation of gradient descent
- [Lecture 03](lecture03.md) - Vanilla and Flux.jl deep neural network implementation
- [Lecture 04](lecture04.md) - Stochastic gradient descent using Flux.jl
- [Lecture 05](lecture05.md) - Filtering, pooling, and convolution
- [Lecture 06](lecture06.md) - Graph convolutional networks (GCNs) for collaborative filtering
- [Lecture 07](lecture07.md) - Autoencoders
- [Lecture 08](lecture08.md) - Deep Q-Learning for the Breakout game
- [Lecture 09](lecture09.md) - Policy gradient methods for the Breakout game

## Course material

```@docs
MachineLearningCourse.MachineLearningCourse
```

