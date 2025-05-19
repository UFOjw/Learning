# Simple Neural Network in NumPy

This project implements a basic fully-connected feedforward neural network using only NumPy. It is designed for educational purposes and demonstrates the core principles of forward and backward propagation, activation functions, and gradient descent optimization.

## Features

* Two-layer neural network
* ReLU activation
* Softmax output layer
* Cross-entropy loss (implied)
* Manual forward and backward pass
* Parameter updates with gradient descent
* Accuracy calculation

## File Overview

* `init_params()` – Initializes weights and biases
* `forward()` – Performs forward propagation
* `ReLU()` and `softmax()` – Activation functions
* `backward()` – Backpropagation to compute gradients
* `update()` – Updates weights and biases using gradients
* `gradient_descent()` – Training loop with logging
* `get_accuracy()` – Evaluates model predictions
* `actual()` – Converts class labels to one-hot format

## Example Usage

```python
# Assuming X is input data with shape (784, number_of_samples)
# and Y is a vector of class labels (0–9) with shape (number_of_samples,)

W1, b1, W2, b2 = gradient_descent(X, Y, learning_rate=0.1, iterations=300)
```
