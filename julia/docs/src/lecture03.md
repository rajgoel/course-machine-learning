# Lecture 03: Deep Neural Networks

This module implements a complete deep neural network from scratch in Julia, including:

- Forward propagation
- Backpropagation
- Gradient descent training
- MNIST digit classification demo

## Architecture

The implementation uses a flexible architecture where you can specify any number of layers:

```julia
# Example architectures:
network_small = DNN([2, 4, 1])           # Simple: 2 → 4 → 1
network_mnist = DNN([784, 128, 64, 10])  # MNIST: 784 → 128 → 64 → 10
network_deep = DNN([100, 50, 25, 10, 1]) # Deep: 100 → 50 → 25 → 10 → 1
```

## Key Features

### Activation Functions
- **Hidden layers**: ReLU activation (σ(z) = max(0, z))
- **Output layer**: Linear activation (for regression/raw scores)

### Loss Function
- Mean Squared Error: ℒ = ||y - ŷ||²

### Initialization
- **Weights**: He initialization (W ~ N(0, 2/n_in))
- **Biases**: Zero initialization

### Training Algorithm
- Standard gradient descent
- Configurable learning rate
- Configurable number of epochs

## MNIST Demo

The demo showcases the neural network on the classic MNIST handwritten digit classification task:

```julia
demo()  # Run with defaults
demo(0.01, 100)  # Custom learning rate and epochs
```

### Demo Process
1. **Data Loading**: Downloads and preprocesses MNIST dataset
2. **Preprocessing**: 
   - Flattens 28×28 images to 784-dimensional vectors
   - Normalizes pixel values to [0, 1]
   - One-hot encodes labels for 10 digit classes
3. **Training**: Trains network using backpropagation
4. **Evaluation**: 
   - Computes overall test accuracy
   - Shows per-digit classification performance
   - Displays sample predictions with confidence scores

### Expected Performance
With default settings, the network typically achieves:
- **Training time**: ~2-3 minutes
- **Test accuracy**: ~85-95% (depending on random initialization)
- **Per-digit performance**: Usually best on digits 0, 1, 6; more challenging on 4, 8, 9

## Mathematical Background

### Forward Propagation
For each layer l:
```
z^[l] = W^[l] * a^[l-1] + b^[l]
a^[l] = σ(z^[l])
```

### Backpropagation
Computing gradients via chain rule:
```
δ^[L] = ∂ℒ/∂a^[L]
δ^[l] = (W^[l+1])^T * δ^[l+1] ⊙ σ'(z^[l])
∂ℒ/∂W^[l] = δ^[l] * (a^[l-1])^T
∂ℒ/∂b^[l] = δ^[l]
```

### Parameter Updates
Standard gradient descent:
```
W^[l] ← W^[l] - α * ∂ℒ/∂W^[l]
b^[l] ← b^[l] - α * ∂ℒ/∂b^[l]
```