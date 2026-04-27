"""
    Lecture05

Filtering, pooling, and convolution.

# Available Functions

- `filter_image()`: Demonstrate image filtering with a given image and kernel
- `stock_data_moving_average()`: Moving average of stock data
- `demo()`: CNN demo for MNIST handwritten digit recognition

# Usage

```julia
using ImageView
using MachineLearningCourse
image = Lecture05.filter_image()
imshow(image)
```

```julia
using MachineLearningCourse
Lecture05.stock_data_moving_average()
```

```julia
using MachineLearningCourse
Lecture05.demo()
```
"""
module Lecture05

# Include all the implementation files
include("SlidingFilter.jl") 
include("ImageFiltering.jl")
include("StockDataMovingAverage.jl")

export filter_image, stock_data_moving_average

include("LeNet5.jl")
include("Demo.jl")

export demo

end
