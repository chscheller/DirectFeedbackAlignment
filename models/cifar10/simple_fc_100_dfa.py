from dataset.cifar10.loader import Cifar10
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution_im2col import Convolution
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDOptimizer, GDMomentumOptimizer

import numpy as np

def simple_fc_100_dfa():

    # Load train data
    cifar = Cifar10('dataset/cifar10')
    X, y = cifar.load_train_set()
    X_test, y_test = cifar.load_test_set()

    # np.random.seed(0)

    layers = [
        MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model = Model(
        layers=layers,
        input_size=(3, 32, 32),
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=0.01, mu=0.9),
        method='dfa',
        regularization=0.0001,
        # lr_decay=0.5,
        # lr_decay_interval=100
    )

    # X = X[0:25, :, :, :]
    # y = y[0:25]

    # model.train(X, y, num_passes=2000, batch_size=20)#
    # model.test(X, y)

    model.train(X, y, num_passes=10, batch_size=64)
    model.test(X_test, y_test)

    return model
