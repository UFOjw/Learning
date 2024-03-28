from matplotlib import pyplot as plt

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, 0)
    return predictions


def test_prediction(img_index, W1, b1, W2, b2, X_train, Y_train):
    current_image = X_train[:, img_index, None]
    prediction = make_predictions(X_train[:, img_index, None], W1, b1, W2, b2)
    label = Y_train[img_index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
