from multiprocessing import freeze_support

import numpy as np

from dataset.cifar10.loader import Cifar10
from dataset.mnist.loader import MNIST
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution import Convolution
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDOptimizer

if __name__ == '__main__':
    freeze_support()

    # Load train data
    mnist = MNIST('dataset/mnist')
    images, labels = mnist.load_training()
    X = np.asarray(images)
    X = X.reshape((X.shape[0], 1, 28, 28))
    y = labels

    images_test, labels_test = mnist.load_training()
    X_test = np.asarray(images_test)
    X_test = X.reshape((X_test.shape[0], 1, 28, 28))
    y_test = labels_test

    #cifar = Cifar10('dataset/cifar10')
    #X, y = cifar.load_train_set()
    #X_test, y_test = cifar.load_test_set()

    layers = [
        # MaxPool(size=2, stride=2),
        # Convolution((1, 1, 3, 3), 1, 1, 0, activation=activation.tanh),
        # MaxPool(size=2, stride=2),
        Convolution((3, 1, 3, 3), 1, 1, 0, activation=activation.tanh),
        Convolution((3, 3, 3, 3), 1, 1, 0.5, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((1, 3, 3, 3), 1, 1, 0.5, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDOptimizer(lr=0.001)
    )

    X_check = np.zeros((1, 1, 28, 28))
    X_check[0,:,:,:] = X[0]
    y_check = y[0]
    # model.gradient_check(X_check, y_check, method='bp')
    model.train(X, y, num_passes=10, batch_size=256, method='bp')