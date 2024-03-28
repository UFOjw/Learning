import numpy as np

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
  
def forward(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def backward(Z1, A1, Z2, A2, W1, W2, X, Y):
    actualY = actual(Y)
    dZ2 = A2 - actualY
    dW2 = 1 / 60000 * dZ2.dot(A1.T)
    db2 = 1 / 60000 * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLUDer(Z1)
    dW1 = 1 / 60000 * dZ1.dot(X.T)
    db1 = 1 / 60000 * np.sum(dZ1)
    return dW1, db1, dW2, db2
  
def ReLUВDer(Z):
    return Z > 0

def actual(Y):
    actualY = np.zeros((Y.max() + 1, Y.size))
    actualY[Y, np.arange(Y.size)] = 1
    return actualY

def update(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 25 == 0:
            print("Iteration: ", i)
            pred = np.argmax(A2, 0)
            print(get_accuracy(pred, Y))
    return W1, b1, W2, b2
