from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters
import scipy.interpolate

import dataset.cifar10_dataset
import dataset.mnist_dataset

from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    """
    """
    freeze_support()

    num_iteration = 20
    data = dataset.cifar10_dataset.load()

    layers = [
        ConvToFullyConnected(),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    # -------------------------------------------------------
    # Train with BP
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
        lr_decay=0.5,
        lr_decay_interval=4
    )

    print("\nRun training:\n------------------------------------")

    stats_shallow = model.train(data_set=data, method='dfa', num_passes=num_iteration, batch_size=64)
    loss, accuracy = model.cost(*data.test_set())

    print("\nResult:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))

    print("\nTrain statisistics:\n------------------------------------")

    print("time spend during forward pass: {}".format(stats_shallow['forward_time']))
    print("time spend during backward pass: {}".format(stats_shallow['backward_time']))
    print("time spend during update pass: {}".format(stats_shallow['update_time']))
    print("time spend in total: {}".format(stats_shallow['total_time']))

    # plt.title('Loss function')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(np.arange(len(stats_bp['train_loss'])), stats_bp['train_loss'])
    # plt.legend(['train loss bp'], loc='best')
    # plt.grid(True)
    # plt.show()

    # plt.title('Accuracy')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.plot(np.arange(len(stats_bp['train_accuracy'])), stats_bp['train_accuracy'])
    # plt.legend(['train accuracy bp'], loc='best')
    # plt.grid(True)
    # plt.show()

    # exit()

    layers = [ConvToFullyConnected()]

    for i in range(10):
        layers += [FullyConnected(size=240, activation=activation.tanh)]

    layers += [FullyConnected(size=10, activation=None, last_layer=True)]

    # -------------------------------------------------------
    # Train with DFA
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
        lr_decay=0.5,
        lr_decay_interval=4
    )

    print("\nRun training:\n------------------------------------")

    stats_deep = model.train(data_set=data, method='dfa', num_passes=num_iteration, batch_size=64)
    loss, accuracy = model.cost(*data.test_set())

    print("\nResult:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))

    print("\nTrain statisistics:\n------------------------------------")

    print("time spend during forward pass: {}".format(stats_deep['forward_time']))
    print("time spend during backward pass: {}".format(stats_deep['backward_time']))
    print("time spend during update pass: {}".format(stats_deep['update_time']))
    print("time spend in total: {}".format(stats_deep['total_time']))

    # plt.title('Loss function')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(np.arange(len(stats_dfa['train_loss'])), stats_dfa['train_loss'])
    # plt.legend(['train loss dfa'], loc='best')
    # plt.grid(True)
    # plt.show()

    # plt.title('Accuracy')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.plot(np.arange(len(stats_dfa['train_accuracy'])), stats_dfa['train_accuracy'])
    # plt.legend(['train accuracy dfa'], loc='best')
    # plt.grid(True)
    # plt.show()

    # exit()

    # train & valid
    plt.title('Loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    shallow_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_shallow['train_loss'], sigma=10)
    deep_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_deep['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats_shallow['train_loss'])), shallow_train_loss)
    plt.plot(stats_shallow['valid_step'], stats_shallow['valid_loss'])
    plt.plot(np.arange(len(stats_deep['train_loss'])), deep_train_loss)
    plt.plot(stats_deep['valid_step'], stats_deep['valid_loss'])
    plt.legend(['1xtanh train loss', '1xtanh validation loss', '10xtanh train loss', '10xtanh  validation loss'], loc='best')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    shallow_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_shallow['train_accuracy'], sigma=10)
    deep_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_deep['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats_shallow['train_accuracy'])), shallow_train_accuracy)
    plt.plot(stats_shallow['valid_step'], stats_shallow['valid_accuracy'])
    plt.plot(np.arange(len(stats_deep['train_accuracy'])), deep_train_accuracy)
    plt.plot(stats_deep['valid_step'], stats_deep['valid_accuracy'])
    plt.legend(['1xtanh  train accuracy', '1xtanh  validation accuracy', '10xtanh  train accuracy', '10xtanh  validation accuracy'], loc='best')
    plt.grid(True)
    plt.show()