## @file main.py
#
# @details Before running main.py, the mnist dataset should be downloaded. Run
# the script data/dataset.py to automatically download the dataset from the
# database.
#
# Run: `python data/dataset.py data/` to download the dataset into data/. Or cd
# into data with `cd data/` and then run `python dataset.py`

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import data.dataset as dataset

def integer_to_positional_vector(i, ndim):
    l = [0.0] * ndim
    l[i] = 1.0
    return np.array(l)

def load_and_prepare_dataset():
    training_images, training_labels, test_images, test_labels = dataset.load("data/mnist.pkl")

    training_images = training_images.astype(np.float)
    test_images = test_images.astype(np.float)
    nn_training_labels = np.ndarray((len(training_labels),10), dtype=float)
    nn_test_labels = np.ndarray((len(training_labels),10), dtype=float)

    for i, v in enumerate(training_images):
        training_images[i] = np.interp(v, (0, 255), (0, 1))

    for i, v in enumerate(training_labels):
        nn_training_labels[i] = integer_to_positional_vector(v, 10)

    for i, v in enumerate(test_images):
        test_images[i] = np.interp(v, (0, 255), (0, 1))
        
    for i, v in enumerate(test_labels):
        nn_test_labels[i] = integer_to_positional_vector(v, 10)

    return training_images, nn_training_labels, test_images, nn_test_labels

if __name__ == "__main__":
    training_images, training_labels, test_images, test_labels = load_and_prepare_dataset()

    model = Sequential()

    # Reasoning about the network topology:
    #
    # The characters 1, 2, 3, 4, 5, 6, 7, 8, 9 and 0 can be decomposed into
    # subfigures like "O | - /". See reasoning.jpeg for the reasoning on paper
    # (pictured).
    #
    # Using this technique 9 different figures can be found. When a character
    # contains a given figure twice, for example 8 which contains two circles,
    # each figure is considered a unique figure.
    #
    # So somewhere in the network topology, this arbitration must be made. A
    # layer containing 9 neurons can make this arbitration. the next layer can
    # combine the figures found by the previous layers and decide what character
    # is seen. This layer can be the output layer as well.
    #
    # However, a figure may be displaced in the picture, with (784)^2
    # possibilities, 784 in each direction. This could be modelled as two layers
    # preceeding the figure finding layer with each 392 neurons  
    model.add(Dense(784, use_bias=True, activation="tanh", input_shape=(784,)))
    model.add(Dense(392, use_bias=True, activation="tanh"))
    model.add(Dense(392, use_bias=True, activation="tanh"))
    model.add(Dense(9, use_bias=True, activation="tanh"))
    model.add(Dense(10, use_bias=True, activation="tanh"))

    model.compile(loss="MSE",
                optimizer="sgd",
                metrics=["accuracy"])

    model.fit(training_images, training_labels, epochs=100, batch_size=32)

    print(model.predict_on_batch(test_images)[0], "=>", test_labels[0])