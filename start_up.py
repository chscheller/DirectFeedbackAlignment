from multiprocessing import freeze_support

import numpy as np
from matplotlib import pyplot as plt
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution import Convolution
from network.layers.dropout import Dropout
from network.layers.max_pool import MaxPool

from dataset.cifar10.loader import Cifar10
from network.layers.fully_connected import FullyConnected
from network.network import Network


def softmax(z) -> np.ndarray:
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def show_image(image: np.ndarray) -> None:
    plt.imshow(image, interpolation='bilinear')
    plt.show()


if __name__ == '__main__':
    freeze_support()

    net = Network([
        Convolution((5, 3, 3, 3), 1, 1),
        Dropout(0.5),
        Convolution((5, 5, 3, 3), 1, 1),
        Dropout(0.5),
        MaxPool(2, 2),
        Convolution((1, 5, 3, 3), 1, 1),
        Dropout(0.5),
        MaxPool(2, 2),
        ConvToFullyConnected(),
        FullyConnected(10, activation=softmax, last_layer=True)
    ])

    # Load train data
    #mnist = MNIST('mnist')
    #images, labels = mnist.load_training()
    #X = np.asarray(images)
    #X = X.reshape((X.shape[0], 1, 28, 28))
    #y = labels
    #
    #images_test, labels_test = mnist.load_training()
    #X_test = np.asarray(images_test)
    #X_test = X.reshape((X_test.shape[0], 1, 28, 28))
    #y_test = labels_test
    cifar = Cifar10('dataset/cifar10')
    X, y = cifar.load_test_set()
    X_test, y_test = cifar.load_test_set()

    # Network params
    train_method = 'dfa'
    num_passes = 10
    learning_rate = 0.0002
    learning_rate_decay = 0.5
    learning_rate_decay_interval = 200
    regularization = 0  #0.01
    batch_size = 512

    net.train(X, y, regularization, learning_rate, num_passes, batch_size, train_method)
    net.test(X_test, y_test)
