from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np

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
    Train statisistics DFA:
    ------------------------------------
    time spend during forward pass: 186.65376925468445
    time spend during backward pass: 196.10941076278687
    time spend during update pass: 144.07492351531982
    time spend in total: 752.406553030014
    
    Train statisistics BP:
    ------------------------------------
    time spend during forward pass: 187.12717700004578
    time spend during backward pass: 331.23116517066956
    time spend during update pass: 152.84005665779114
    time spend in total: 911.1534883975983
    """
    freeze_support()

    num_iteration = 20

    data = dataset.cifar10_dataset.load()

    layers = [
        ConvToFullyConnected(),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    # -------------------------------------------------------
    # Train with DFA
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-4, mu=0.9),
        regularization=0.001,
        lr_decay=0.5,
        lr_decay_interval=4
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

    # -------------------------------------------------------
    # Train with BP
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
        regularization=0.001,
        lr_decay=0.5,
        lr_decay_interval=4
    )

    print("\nRun training:\n------------------------------------")

    stats_bp = model.train(data_set=data, method='bp', num_passes=num_iteration, batch_size=64)
    loss, accuracy = model.cost(*data.test_set())

    print("\nResult:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))

    print("\nTrain statisistics:\n------------------------------------")

    print("time spend during forward pass: {}".format(stats_bp['forward_time']))
    print("time spend during backward pass: {}".format(stats_bp['backward_time']))
    print("time spend during update pass: {}".format(stats_bp['update_time']))
    print("time spend in total: {}".format(stats_bp['total_time']))

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(len(stats_dfa['train_loss'])), stats_dfa['train_loss'])
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_loss'])
    plt.plot(np.arange(len(stats_bp['train_loss'])), stats_bp['train_loss'])
    plt.plot(stats_bp['valid_step'], stats_bp['valid_loss'])
    plt.legend(['train loss dfa', 'validation loss dfa', 'train loss bp', 'validation loss bp'], loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])), stats_dfa['train_accuracy'])
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_accuracy'])
    plt.plot(np.arange(len(stats_bp['train_accuracy'])), stats_bp['train_accuracy'])
    plt.plot(stats_bp['valid_step'], stats_bp['valid_accuracy'])
    plt.legend(['train accuracy dfa', 'validation accuracy dfa', 'train accuracy bp', 'validation accuracy bp'], loc='lower right')
    plt.grid(True)
    plt.show()