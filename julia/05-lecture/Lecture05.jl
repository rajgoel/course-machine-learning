"""
# Lecture05: Convolutional Neural Networks

Implementation of LeNet-5 CNN architecture and sliding filter operations.
Demonstrates convolutional layers, pooling, and practical applications.

## Exported Functions

### CNN Architecture
- [`FluxLeNet5`](@ref): LeNet-5 CNN constructor for image classification
- [`train!`](@ref): CNN training with mini-batch SGD
- [`predict`](@ref): CNN inference on image data
- [`accuracy`](@ref): Model accuracy evaluation
- [`evaluate`](@ref): Comprehensive model evaluation with confusion matrix

### Sliding Filters
- [`sliding_filter`](@ref): Generic N-dimensional sliding filter operation
- [`apply_filter`](@ref): Image filtering with predefined kernels
- [`kernels`](@ref): Dictionary of common image processing kernels

### Data Loading
- [`load_svhn_raw`](@ref): Load raw SVHN street view house numbers dataset
- [`load_svhn_data`](@ref): Load preprocessed SVHN data for training

### Signal Processing
- [`plot_stock_with_moving_average`](@ref): Stock price visualization with moving averages

## Usage Examples

Create and train a LeNet-5 CNN:

```julia
using MachineLearningCourse.Lecture05

# Create CNN for MNIST (28×28 grayscale images)
network = FluxLeNet5([28, 28, 1])

# Train on image data (height × width × channels × samples)
losses = train!(network, X_train, Y_train, 0.001, 50; batch_size=128)

# Evaluate on test set
results = evaluate(network, X_test, Y_test, 10)
println("Accuracy: \$(results.accuracy)")
```

## Running Demos

Demo functions are not exported but can be accessed directly:

```julia
using MachineLearningCourse.Lecture05

# CNN demos
Lecture05.demo_lenet5_mnist()  # LeNet-5 on MNIST dataset
Lecture05.demo_lenet5_svhn()   # LeNet-5 on SVHN dataset

# Image filtering demo
Lecture05.demo_image_filtering()

# Stock moving average demo  
Lecture05.demo_moving_average()
```

Access utility functions directly:

```julia
# Image filters and kernels
blurred = Lecture05.apply_filter(image, Lecture05.kernels[:gaussian_blur])
edges = Lecture05.apply_filter(image, Lecture05.kernels[:sobel_x])

# Custom sliding filter
custom_kernel = [1 0 -1; 1 0 -1; 1 0 -1] / 3
filtered = Lecture05.sliding_filter(data, custom_kernel; operation=sum)

# Stock visualization
plot = Lecture05.plot_stock_with_moving_average("AAPL", 20, "1y")
```
"""
module Lecture05

# Export core CNN functions for public use
export FluxLeNet5, train!, predict, accuracy, evaluate

# Include all the implementation files
include("sliding_filter.jl") 
include("image_filtering.jl")
include("moving_average.jl")
include("../04-lecture/mnist_data.jl")
include("FluxLeNet5.jl")
include("svhn_data.jl")
include("Demo_MNIST.jl")
include("Demo_SVHN.jl") 

end
