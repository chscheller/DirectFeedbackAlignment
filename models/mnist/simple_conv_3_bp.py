import numpy as np

from dataset.mnist.loader import MNIST
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution_im2col import Convolution
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDOptimizer, GDMomentumOptimizer


def simple_conv_3_bp():
    """
    """

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

    X = X / 255
    X_test = X_test / 255

    layers = [
        Convolution((3, 1, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        Convolution((3, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((3, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model = Model(
        layers=layers,
        input_size=(1, 28, 28),
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=0.01, mu=0.9),
        method='bp',
        regularization=0.0001
    )

    X = X[0:25,:,:,:]
    y = y[0:25]

    model.train(X, y, num_passes=100, batch_size=20)
    model.test(X_test, y_test)

    return model
