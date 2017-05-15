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
    freeze_support()

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
        Convolution((32, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=4, stride=2),
        Convolution((32, 32, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=3, stride=2),
        Convolution((64, 32, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=3, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=64, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-4, mu=0.9),
        regularization=0.001,
        # lr_decay=0.5,
        # lr_decay_interval=100
    )

    print("\nRun training:\n------------------------------------")

    stats = model.train(data_set=data, method='dfa', num_passes=2, batch_size=100)
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
    plt.plot(np.arange(len(stats['train_loss'])), stats['train_loss'])
    plt.plot(stats['valid_step'], stats['valid_loss'])
    plt.legend(['train loss', 'validation loss'], loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(np.arange(len(stats['train_accuracy'])), stats['train_accuracy'])
    plt.plot(stats['valid_step'], stats['valid_accuracy'])
    plt.legend(['train accuracy', 'validation accuracy'], loc='upper right')
    plt.grid(True)
    plt.show()