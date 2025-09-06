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

# Run Lecture 05 LeNet-5 CNN demo
MachineLearningCourse.Lecture05.demo_lenet5()

# Run Lecture 05 image filtering demo
MachineLearningCourse.Lecture05.demo_image_filtering("path/to/image.jpg")

# Run Lecture 05 stock moving average demo
MachineLearningCourse.Lecture05.demo_moving_average()
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

### Lecture 05 - Convolutional Neural Networks and Filters

This lecture covers:
- **LeNet-5 CNN Architecture**: Implementation of the classic LeNet-5 convolutional neural network for MNIST digit classification
- **Image Filtering**: Sliding filter operations for edge detection, blurring, sharpening, and other image processing effects
- **Moving Averages**: Stock market data analysis with moving average filters using real-time financial data

Key features:
- Complete LeNet-5 implementation with Flux.jl
- Generic sliding filter function for N-dimensional data
- Interactive stock market analysis with Dow 30 companies
- Image processing with multiple filter kernels (Sobel, Laplacian, Gaussian blur, etc.)

```@autodocs
Modules = [MachineLearningCourse.Lecture05]
```

