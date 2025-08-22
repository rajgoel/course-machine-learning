# File: julia/03-lecture/Lecture03.jl
# Lecture 3 submodule: Deep Neural Networks

"""
Lecture03: Deep Neural Networks

A vanilla implementation of deep neural networks with fully connected
layers, ReLU activation for all hidden layers, and MSE loss.

Main exports:
- DNN: Deep Neural Network structure
- train!: Training function  
- predict: Prediction function
- demo: Trains on MNIST hand-written digit recognition

Example usage:
  using MachineLearningCourse.Lecture03
  
  # Create network with: 2 inputs → 4 hidden → 3 hidden → 1 output
  network = DNN([2, 4, 3, 1])
    
  # Train on data  with learning rate 0.1 and 1000 epochs
  losses = train!(network, X_train, Y_train, 0.1, 1000)
    
  # Make predictions
  prediction = predict(network, x_test)
    
  # Run the demo with default parameter (Learning rate: 0.001, epochs: 50)
  demo()

  # Run the demo with learning rate: 0.005, epochs: 500)
  demo(0.005,500)
"""
module Lecture03

# Include the implementation files
include("DNN.jl")
include("Demo.jl")

# Essential public API:
export DNN, train!, predict
# Essential public API:
export demo

end # module Lecture03
