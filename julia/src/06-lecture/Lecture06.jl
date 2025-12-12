"""
    Lecture06

Implementation of graph convolutional networks (GCNs) for collaborative filtering.

# Available Functions

- `demo()`: Graph convolutional network demo for collaborative filtering using MovieLens data

# Usage

```julia
using MachineLearningCourse
Lecture06.demo()
```
"""
module Lecture06

# Include all the implementation files
include("Graph.jl")
include("MovieLens_data.jl")
include("GCN.jl")
include("Demo.jl")

export demo, movie_explorer

end
