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
    
    
    
    ------------------------------------
    DFA
    ------------------------------------
    Result:
    ------------------------------------
    loss on test set: 1.5317771647628007
    accuracy on test set: 0.4463
    
    Train statisistics:
    ------------------------------------
    time spend during forward pass: 621.903642654419
    time spend during backward pass: 90.21121907234192
    time spend during update pass: 1.5559537410736084
    time spend in total: 788.1902389526367

    ------------------------------------
    BP
    ------------------------------------
    Result:
    ------------------------------------
    loss on test set: 1.0581702546107525
    accuracy on test set: 0.6258
    
    Train statisistics:
    ------------------------------------
    time spend during forward pass: 575.9978523254395
    time spend during backward pass: 269.75944471359253
    time spend during update pass: 1.299713134765625
    time spend in total: 918.6962807178497
    """

    freeze_support()

    num_iteration = 20

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
        MaxPool(size=2, stride=2),
        Convolution((16, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((16, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((32, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=64, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    # -------------------------------------------------------
    # Train with BP
    # -------------------------------------------------------

    # model = Model(
    #     layers=layers,
    #     num_classes=10,
    #     optimizer=GDMomentumOptimizer(lr=1e-2, mu=0.9),
    #     regularization=0.001,
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

    # plt.title('Loss function')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(np.arange(len(stats_bp['train_loss'])), stats_bp['train_loss'])
    # plt.plot(stats_bp['valid_step'], stats_bp['valid_loss'])
    # plt.legend(['train loss bp', 'validation loss bp'], loc='upper right')
    # plt.grid(True)
    # plt.show()
#
    # plt.title('Accuracy')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.plot(np.arange(len(stats_bp['train_accuracy'])), stats_bp['train_accuracy'])
    # plt.plot(stats_bp['valid_step'], stats_bp['valid_accuracy'])
    # plt.legend(['train accuracy bp', 'validation accuracy bp'], loc='lower right')
    # plt.grid(True)
    # plt.show()

    # -------------------------------------------------------
    # Train with DFA
    # -------------------------------------------------------

    model = Model(
        # layers=layers,
        # num_classes=10,
        # optimizer=GDMomentumOptimizer(lr=1.5*1e-2, mu=0.9),
        # # regularization=0.001,
        # lr_decay=0.5,
        # lr_decay_interval=3
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=5*1e-3, mu=0.9),
        # regularization=0.001,
        # lr_decay=0.5,
        # lr_decay_interval=3
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
    plt.plot(np.arange(len(stats_dfa['train_loss'])), stats_dfa['train_loss'])
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_loss'])
    plt.legend(['train loss dfa', 'validation loss dfa'], loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(np.arange(len(stats_dfa['train_accuracy'])), stats_dfa['train_accuracy'])
    plt.plot(stats_dfa['valid_step'], stats_dfa['valid_accuracy'])
    plt.legend(['train accuracy dfa', 'validation accuracy dfa'], loc='lower right')
    plt.grid(True)
    plt.show()

    exit()

    # plt.title('Loss function')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
    # bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
    # plt.plot(np.arange(len(stats_dfa['train_loss'])), bp_train_loss)
    # plt.plot(stats_dfa['valid_step'], stats_dfa['valid_loss'])
    # plt.plot(np.arange(len(stats_bp['train_loss'])), bp_train_loss)
    # plt.plot(stats_bp['valid_step'], stats_bp['valid_loss'])
    # plt.legend(['train loss dfa', 'validation loss dfa', 'train loss bp', 'validation loss bp'], loc='upper right')
    # plt.grid(True)
    # plt.show()

    # plt.title('Accuracy')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
    # bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
    # plt.plot(np.arange(len(stats_dfa['train_accuracy'])), dfa_train_accuracy)
    # plt.plot(stats_dfa['valid_step'], stats_dfa['valid_accuracy'])
    # plt.plot(np.arange(len(stats_bp['train_accuracy'])), bp_train_accuracy)
    # plt.plot(stats_bp['valid_step'], stats_bp['valid_accuracy'])
    # plt.legend(['train accuracy dfa', 'validation accuracy dfa', 'train accuracy bp', 'validation accuracy bp'], loc='lower right')
    # plt.grid(True)
    # plt.show()
