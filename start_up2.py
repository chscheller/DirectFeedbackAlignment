from multiprocessing import freeze_support

import numpy as np

from dataset.cifar10.loader import Cifar10
from dataset.mnist.loader import MNIST
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution import Convolution
from network.layers.dropout import Dropout
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDOptimizer

if __name__ == '__main__':
    freeze_support()

    # Load train data
    cifar = Cifar10('dataset/cifar10')
    X, y = cifar.load_train_set()
    X_test, y_test = cifar.load_test_set()

    layers = [
        #MaxPool(size=2, stride=2),
        #Convolution((1, 1, 3, 3), 1, 1, 0, activation=activation.tanh),
        #Convolution((1, 1, 3, 3), 1, 1, 0, activation=activation.tanh),
        #Convolution((1, 1, 3, 3), 1, 1, 0, activation=activation.tanh),
        #Convolution((1, 1, 3, 3), 1, 1, 0.5, activation=activation.tanh),
        #MaxPool(size=2, stride=2),
        #ConvToFullyConnected(),
        #FullyConnected(size=10, activation=None, last_layer=True)

        #Convolution((3, 1, 3, 3), 1, 1, 0, activation=activation.tanh),
        #Convolution((3, 3, 3, 3), 1, 1, 0, activation=activation.tanh),

        Convolution((96, 3, 3, 3), stride=1, padding=1, dropout_rate=0.25, activation=activation.tanh),
        MaxPool(size=4, stride=2),
        Convolution((128, 96, 5, 5), stride=1, padding=1, dropout_rate=0.25, activation=activation.tanh),
        MaxPool(size=3, stride=2),
        Convolution((256, 128, 5, 5), stride=1, padding=1, dropout_rate=0.25, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=2048, dropout_rate=0.5, activation=None, last_layer=True),
        FullyConnected(size=2048, dropout_rate=0.5, activation=None, last_layer=True)
    ]

    model = Model(
        layers=layers,
        input_size=(3, 32, 32),
        num_classes=10,
        optimizer=GDOptimizer(lr=0.2),
        method='bp',
        regularization=0
    )

    X = X[0:25,:,:,:]
    y = y[0:25]

    #X_check = np.zeros((1, 1, 28, 28))
    #X_check[0,:,:,:] = X[0]
    #y_check = y[0]
    #model.gradient_check(X_check, y_check)

    print('cost: {}'.format(model.cost(X[0:1000], y[0:1000])))

    model.train(X, y, num_passes=1000, batch_size=20)
    model.test(X_test, y_test)