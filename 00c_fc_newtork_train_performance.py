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

    num_iteration = 30
    data = dataset.cifar10_dataset.load()
    # data = dataset.mnist_dataset.load()

    layers = [
        ConvToFullyConnected(),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    # # -------------------------------------------------------
    # # Train with BP
    # # -------------------------------------------------------

    # model = Model(
    #     layers=layers,
    #     num_classes=10,
    #     optimizer=GDMomentumOptimizer(lr=1e-2, mu=0.9),
    #     lr_decay=0.5,
    #     lr_decay_interval=5
    # )

    # print("\nRun training:\n------------------------------------")

    # stats_bp = model.train(data_set=data, method='bp', num_passes=num_iteration, batch_size=64)
    # loss, accuracy = model.cost(*data.test_set())

    # print("\nResult:\n------------------------------------")
    # print('loss on test set: {}'.format(loss))
    # print('accuracy on test set: {}'.format(accuracy))

    # print("\nTrain statisistics:\n------------------------------------")

    # print("time spend during forward pass: {}".format(stats_bp['forward_time']))
    # print("time spend during backward pass: {}".format(stats_bp['backward_time']))
    # print("time spend during update pass: {}".format(stats_bp['update_time']))
    # print("time spend in total: {}".format(stats_bp['total_time']))

    # # plt.title('Loss function')
    # # plt.xlabel('epoch')
    # # plt.ylabel('loss')
    # # plt.plot(np.arange(len(stats_bp['train_loss'])), stats_bp['train_loss'])
    # # plt.legend(['train loss bp'], loc='best')
    # # plt.grid(True)
    # # plt.show()

    # # plt.title('Accuracy')
    # # plt.xlabel('epoch')
    # # plt.ylabel('accuracy')
    # # plt.plot(np.arange(len(stats_bp['train_accuracy'])), stats_bp['train_accuracy'])
    # # plt.legend(['train accuracy bp'], loc='best')
    # # plt.grid(True)
    # # plt.show()

    # # exit()

    # -------------------------------------------------------
    # Train with DFA
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
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
    train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_loss'])), train_loss)
    plt.legend(['train loss'], loc='best')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])), train_accuracy)
    plt.legend(['train accuracy'], loc='best')
    plt.grid(True)
    plt.show()

    # plt.title('Loss vs epoch')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # # plt.plot(np.arange(len(stats_dfa['train_loss'])), stats_dfa['train_loss'])
    # # plt.plot(np.arange(len(stats_bp['train_loss'])), stats_bp['train_loss'])
    # dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    # bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
    # plt.plot(np.arange(len(stats_dfa['train_loss'])), dfa_train_loss)
    # plt.plot(np.arange(len(stats_bp['train_loss'])), bp_train_loss)
    # plt.legend(['train loss dfa', 'train loss bp'], loc='best')
    # plt.grid(True)
    # plt.show()

    # plt.title('Accuracy vs epoch')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # # plt.plot(np.arange(len(stats_dfa['train_accuracy'])), stats_dfa['train_accuracy'])
    # # plt.plot(np.arange(len(stats_bp['train_accuracy'])), stats_bp['train_accuracy'])
    # dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    # bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
    # plt.plot(np.arange(len(stats_dfa['train_accuracy'])), dfa_train_accuracy)
    # plt.plot(np.arange(len(stats_bp['train_accuracy'])), bp_train_accuracy)
    # plt.legend(['train accuracy dfa', 'train accuracy bp'], loc='lower right')
    # plt.grid(True)
    # plt.show()

    # # Forward, regularization, update and validation passes are excactly the same operations for dfa and bp. Therefore
    # # they should take euqally long. To ensure that inequalities don't affect the result, we normalize the time here.
    # # The reference time is the one measured for bp.
    # total_time_bp = stats_bp['total_time']
    # total_time_dfa = total_time_bp - stats_bp['backward_time'] + stats_dfa['backward_time']
    # step_to_time_bp = total_time_bp / len(stats_bp['train_loss'])
    # step_to_time_dfa = step_to_time_bp * total_time_dfa / stats_bp['total_time']

    # plt.title('Loss vs time')
    # plt.xlabel('time')
    # plt.ylabel('loss')
    # # plt.plot(np.arange(len(stats_dfa['train_loss'])) * step_to_time_dfa, stats_dfa['train_loss'])
    # # plt.plot(np.arange(len(stats_bp['train_loss'])) * step_to_time_bp, stats_bp['train_loss'])
    # dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    # bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
    # plt.plot(np.arange(len(stats_dfa['train_loss'])) * step_to_time_dfa, dfa_train_loss)
    # plt.plot(np.arange(len(stats_bp['train_loss'])) * step_to_time_bp, bp_train_loss)
    # plt.legend(['train loss dfa', 'train loss bp'], loc='best')
    # plt.grid(True)
    # plt.show()

    # plt.title('Accuracy vs time')
    # plt.xlabel('time')
    # plt.ylabel('accuracy')
    # # plt.plot(np.arange(len(stats_dfa['train_accuracy'])) * step_to_time_dfa, stats_dfa['train_accuracy'])
    # # plt.plot(np.arange(len(stats_bp['train_accuracy'])) * step_to_time_bp, stats_bp['train_accuracy'])
    # dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    # bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
    # plt.plot(np.arange(len(stats_dfa['train_accuracy'])) * step_to_time_dfa, dfa_train_accuracy)
    # plt.plot(np.arange(len(stats_bp['train_accuracy'])) * step_to_time_bp, bp_train_accuracy)
    # plt.legend(['train accuracy dfa', 'train accuracy bp'], loc='lower right')
    # plt.grid(True)
    # plt.show()
