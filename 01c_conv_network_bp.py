from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage.filters

import dataset.cifar10_dataset

from network import activation, weight_initializer
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution_im2col import Convolution
from network.layers.dropout import Dropout
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    """
    """

    freeze_support()

    num_iteration = 20
    data = dataset.cifar10_dataset.load()

    layers = [
        # MaxPool(size=2, stride=2),
        Convolution((8, 3, 4, 4), stride=2, padding=2, dropout_rate=0, activation=activation.tanh),
        #MaxPool(size=2, stride=2),
        Convolution((16, 8, 3, 3), stride=2, padding=1, dropout_rate=0, activation=activation.tanh),
        #MaxPool(size=2, stride=2),
        Convolution((32, 16, 3, 3), stride=2, padding=1, dropout_rate=0, activation=activation.tanh),
        #MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=64, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    # -------------------------------------------------------
    # Train with BP
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=3*1e-2, mu=0.9),
    )

    print("\nRun training:\n------------------------------------")

    stats = model.train(data_set=data, method='bp', num_passes=num_iteration, batch_size=64)
    loss, accuracy = model.cost(*data.test_set())

    print("\nResult:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))

    print("\nTrain statisistics:\n------------------------------------")

    print("time spend during forward pass: {}".format(stats['forward_time']))
    print("time spend during backward pass: {}".format(stats['backward_time']))
    print("time spend during update pass: {}".format(stats['update_time']))
    print("time spend in total: {}".format(stats['total_time']))

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    train_loss = scipy.ndimage.filters.gaussian_filter1d(stats['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats['train_loss'])), train_loss)
    plt.plot(stats['valid_step'], stats['valid_loss'])
    plt.legend(['train loss bp', 'validation loss bp'], loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats['train_accuracy'])), train_accuracy)
    plt.plot(stats['valid_step'], stats['valid_accuracy'])
    plt.legend(['train accuracy bp', 'validation accuracy bp'], loc='lower right')
    plt.grid(True)
    plt.show()