from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters
import scipy.interpolate

import dataset.cifar10_dataset

from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    """
    Goal: Compare DFA and BP training performances with respect to train loss, train accuracy and 
    training time on a fully connected NN
    
    Initial learning rate and learning rate decay parameters were evaluated  by hand by comparing the training 
    performance on the training set for various 
    parameter combinations
    """
    freeze_support()

    num_iteration = 35
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
    # Train with BP
    # -------------------------------------------------------

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1.5*1e-3, mu=0.9),
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

    # train only
    plt.title('Loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_loss'])), dfa_train_loss)
    plt.plot(np.arange(len(stats_bp['train_loss'])), bp_train_loss)
    plt.legend(['train loss dfa', 'train loss bp'], loc='best')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])), dfa_train_accuracy)
    plt.plot(np.arange(len(stats_bp['train_accuracy'])), bp_train_accuracy)
    plt.legend(['train accuracy dfa', 'train accuracy bp'], loc='lower right')
    plt.grid(True)
    plt.show()

    # train & valid
    plt.title('Loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_loss'])), dfa_train_loss)
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_loss'])
    plt.plot(np.arange(len(stats_bp['train_loss'])), bp_train_loss)
    plt.plot(stats_bp['valid_step'], stats_bp['valid_loss'])
    plt.legend(['train loss dfa', 'validation loss dfa', 'train loss bp', 'validation loss bp'], loc='best')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])), dfa_train_accuracy)
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_accuracy'])
    plt.plot(np.arange(len(stats_bp['train_accuracy'])), bp_train_accuracy)
    plt.plot(stats_bp['valid_step'], stats_bp['valid_accuracy'])
    plt.legend(['train accuracy dfa', 'validation accuracy dfa', 'train accuracy bp', 'validation accuracy bp'], loc='best')
    plt.grid(True)
    plt.show()

    # Forward, regularization, update and validation passes are excactly the same operations for dfa and bp. Therefore
    # they should take euqally long. To ensure that inequalities don't affect the result, we normalize the time here.
    # The reference time is the one measured for bp.
    total_time_bp = stats_bp['total_time']
    total_time_dfa = total_time_bp - stats_bp['backward_time'] + stats_dfa['backward_time']
    step_to_time_bp = total_time_bp / len(stats_bp['train_loss'])
    step_to_time_dfa = step_to_time_bp * total_time_dfa / stats_bp['total_time']

    plt.title('Loss vs time')
    plt.xlabel('time')
    plt.ylabel('loss')
    dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_loss'])) * step_to_time_dfa, dfa_train_loss)
    plt.plot(np.arange(len(stats_bp['train_loss'])) * step_to_time_bp, bp_train_loss)
    plt.legend(['train loss dfa', 'train loss bp'], loc='best')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy vs time')
    plt.xlabel('time')
    plt.ylabel('accuracy')
    dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])) * step_to_time_dfa, dfa_train_accuracy)
    plt.plot(np.arange(len(stats_bp['train_accuracy'])) * step_to_time_bp, bp_train_accuracy)
    plt.legend(['train accuracy dfa', 'train accuracy bp'], loc='lower right')
    plt.grid(True)
    plt.show()

    plt.title('Loss vs time')
    plt.xlabel('time')
    plt.ylabel('loss')
    dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_loss'])) * step_to_time_dfa, dfa_train_loss)
    plt.plot(np.asarray(stats_dfa['valid_step']) * step_to_time_dfa, stats_dfa['valid_loss'])
    plt.plot(np.arange(len(stats_bp['train_loss'])) * step_to_time_bp, bp_train_loss)
    plt.plot(np.asarray(stats_bp['valid_step']) * step_to_time_bp, stats_bp['valid_loss'])
    plt.legend(['train loss dfa', 'validation loss dfa', 'train loss bp', 'validation loss bp'], loc='best')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy vs time')
    plt.xlabel('time')
    plt.ylabel('accuracy')
    dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])) * step_to_time_dfa, dfa_train_accuracy)
    plt.plot(np.asarray(stats_dfa['valid_step']) * step_to_time_dfa, stats_dfa['valid_accuracy'])
    plt.plot(np.arange(len(stats_bp['train_accuracy'])) * step_to_time_bp, bp_train_accuracy)
    plt.plot(np.asarray(stats_bp['valid_step']) * step_to_time_bp, stats_bp['valid_accuracy'])
    plt.legend(['train accuracy dfa', 'validation accuracy dfa', 'train accuracy bp', 'validation accuracy bp'], loc='lower right')
    plt.grid(True)
    plt.show()
