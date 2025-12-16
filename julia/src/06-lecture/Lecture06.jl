"""
    Lecture06

Implementation of graph convolutional networks (GCNs) for collaborative filtering.

# Available Functions

- `demo()`: Graph convolutional network demo for collaborative filtering using MovieLens data
- `movie_explorer(embeddings, data)`: Interactive menu for exploring movie similarities using trained embeddings.

# Usage

```julia
using MachineLearningCourse
Lecture06.demo()
```
or
```julia
_, embeddings, data = Lecture06.demo(interactive=false)
Lecture06.movie_explorer(embeddings, data)
```
"""
module Lecture06

# Include all the implementation files
include("Graph.jl")
include("MovieLens_data.jl")
include("GCN.jl")
include("Demo.jl")

export load_movielens_data, demo, movie_explorer

end
