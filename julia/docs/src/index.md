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

### Lecture 01 - A simple linear classifier

```@autodocs
Modules = [MachineLearningCourse.Lecture01]
```

### Lecture 02 - Vanilla implementation of gradient descent

```@autodocs
Modules = [MachineLearningCourse.Lecture02]
```

### Lecture 03 - Vanilla and Flux.jl deep neural network implementation

```@autodocs
Modules = [MachineLearningCourse.Lecture03]
```

### Lecture 04 - Deep neural network implementation using Flux.jl

```@autodocs
Modules = [MachineLearningCourse.Lecture04]
```

### Lecture 05 - Implementation of filtering, pooling, and convolution

```@autodocs
Modules = [MachineLearningCourse.Lecture05]
```

### Lecture 06 - Implementation of graph convolutional networks (GCNs) for collaborative filtering.

```@autodocs
Modules = [MachineLearningCourse.Lecture06]
```

### Lecture 07 - Implementation of autoencoders

```@autodocs
Modules = [MachineLearningCourse.Lecture07]
```

### Lecture 08 - Implementation of Deep Q-Learning for the Breakout game

```@autodocs
Modules = [MachineLearningCourse.Lecture08]
```

### Lecture 09 - Implementation of policy gradient methods for the Breakout game

```@autodocs
Modules = [MachineLearningCourse.Lecture09]
```

