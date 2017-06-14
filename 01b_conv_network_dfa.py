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

    num_iteration = 200
    data = dataset.cifar10_dataset.load()

    # layers = [
    #     Convolution((32, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
    #     Convolution((32, 32, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
    #     MaxPool(size=2, stride=2),
    #     Dropout(0.2),
    #     Convolution((64, 32, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
    #     MaxPool(size=2, stride=2),
    #     Dropout(0.3),
    #     Convolution((128, 64, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
    #     MaxPool(size=2, stride=2),
    #     Dropout(0.4),
    #     ConvToFullyConnected(),
    #     FullyConnected(size=512, activation=activation.tanh),
    #     FullyConnected(size=10, activation=None, last_layer=True)
    # ]

    layers = [
        Convolution((8, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((16, 8, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((16, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((32, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=64, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

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
    # Train with DFA
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-2, mu=0.9),
        lr_decay=0.5,
        lr_decay_interval=50
    )

    print("\nRun training:\n------------------------------------")

    stats_dfa = model.train(data_set=data, method='dfa', num_passes=num_iteration, batch_size=64)
    loss, accuracy = model.cost(*data.test_set())

    print("\nResult:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))

    print("\nTrain statisistics:\n------------------------------------")

    print("time spend during forward pass: {}".format(stats_dfa['forward_time']))
    print("time spend during backward pass: {}".format(stats_dfa['backward_time']))
    print("time spend during update pass: {}".format(stats_dfa['update_time']))
    print("time spend in total: {}".format(stats_dfa['total_time']))

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_loss'])), dfa_train_loss)
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_loss'])
    plt.legend(['train loss dfa', 'validation loss dfa'], loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])), dfa_train_accuracy)
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_accuracy'])
    plt.legend(['train accuracy dfa', 'validation accuracy dfa'], loc='lower right')
    plt.grid(True)
    plt.show()