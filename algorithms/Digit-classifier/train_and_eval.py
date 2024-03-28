from model import *
from prediction import *

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (60000, 784)).T / 255
x_test = np.reshape(x_test, (10000, 784)).T / 255

W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, 500)

test_prediction(0, W1, b1, W2, b2, x_test, y_test)
test_prediction(1, W1, b1, W2, b2, x_test, y_test)
test_prediction(2, W1, b1, W2, b2, x_test, y_test)
test_prediction(3, W1, b1, W2, b2, x_test, y_test)

predictions = make_predictions(x_test, W1, b1, W2, b2)
print(get_accuracy(predictions, y_test))
