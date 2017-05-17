from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np

import dataset.mnist_dataset
import dataset.cifar10_dataset

from network import activation, weight_initializer
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    freeze_support()

    num_hidden_units = 240

    data = dataset.mnist_dataset.load('dataset/mnist')
    # data = dataset.cifar10_dataset.load()

    initializers = [
        weight_initializer.RandomUniform(-1, 1),
        # weight_initializer.RandomNormal(-1, 0),
        weight_initializer.RandomUniform(0, 1),
        weight_initializer.RandomUniform(0, 2),
    ]

    lrs = [
        # 1e-1,
        1e-2,
        1e-3,
        # 1e-4
    ]

    statistics = []

    for lr in lrs:
        curr_stats = []
        for initializer in initializers:
            layers = [
                ConvToFullyConnected(),
                FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer),
                FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer),
                FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer),
                FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer),
                FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer),
                FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer),
                FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer),
                FullyConnected(size=10, activation=None, last_layer=True)
            ]

            model = Model(
                layers=layers,
                num_classes=10,
                optimizer=GDMomentumOptimizer(lr=lr, mu=0.9),
                # regularization=0.001,
                # lr_decay=0.5,
                # lr_decay_interval=100
            )

            print("\n\n------------------------------------")

            print("Initialize: {}".format(initializer))

            print("\nRun training:\n------------------------------------")

            stats = model.train(data_set=data, method='dfa', num_passes=2, batch_size=50)
            loss, accuracy = model.cost(*data.test_set())

            print("\nResult:\n------------------------------------")
            print('loss on test set: {}'.format(loss))
            print('accuracy on test set: {}'.format(accuracy))
            curr_stats.append(stats)

        statistics.append(curr_stats)

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    labels = []
    for j in range(len(lrs)):
        for i in range(len(initializers)):
            stats = statistics[j][i]
            plt.plot(np.arange(len(stats['train_loss'])), stats['train_loss'])
            # plt.plot(stats['valid_step'], stats['valid_loss'])
            labels.append("{}, lr={}: train loss".format(initializers[i], lrs[j]))
            # labels.append("{}, lr={}: validation loss".format(initializers[i], lrs[j]))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for j in range(len(lrs)):
        for i in range(len(initializers)):
            stats = statistics[j][i]
            plt.plot(np.arange(len(stats['train_accuracy'])), stats['train_accuracy'])
            # plt.plot(stats['valid_step'], stats['valid_accuracy'])
            labels.append("{}, lr={}: train accuracy".format(initializers[i], lrs[j]))
            # labels.append("{}, lr={}: validation accuracy".format(initializers[i], lrs[j]))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()