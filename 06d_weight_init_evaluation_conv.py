from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np

import dataset.mnist_dataset
import dataset.cifar10_dataset

from network import activation, weight_initializer
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution_im2col import Convolution
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    freeze_support()

    num_hidden_units = 240

    # data = dataset.mnist_dataset.load('dataset/mnist')
    data = dataset.cifar10_dataset.load()

    # initializers = [
    #     # weight_initializer.Fill(0),
    #     # weight_initializer.Fill(1e-3),
    #     # weight_initializer.Fill(1),
    #     # weight_initializer.Fill(100),
    #     # weight_initializer.RandomUniform(-1, 1),
    #     # weight_initializer.RandomUniform(-1/np.sqrt(num_hidden_units), 1/np.sqrt(num_hidden_units)),
    #     # weight_initializer.RandomUniform(-1/num_hidden_units, 1/num_hidden_units),
    #     # weight_initializer.RandomUniform(-100, 100),
    #     # weight_initializer.RandomNormal(1, 0),
    #     weight_initializer.RandomNormal(1/np.sqrt(num_hidden_units)),
    #     weight_initializer.RandomNormal(3/np.sqrt(num_hidden_units)),
    #     weight_initializer.RandomNormal(1/(3 * np.sqrt(num_hidden_units))),
    # ]

    initializers = ['Normal(1, 0)', 'Normal(1/sqrt(fan_out), 0)']

    model_layers = [
        [
            MaxPool(size=2, stride=2),
            Convolution((16, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh,
                        weight_initializer=weight_initializer.Fill(0), fb_weight_initializer=weight_initializer.RandomNormal()),
            MaxPool(size=2, stride=2),
            Convolution((16, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh,
                        weight_initializer=weight_initializer.Fill(0), fb_weight_initializer=weight_initializer.RandomNormal()),
            MaxPool(size=2, stride=2),
            Convolution((32, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh,
                        weight_initializer=weight_initializer.Fill(0), fb_weight_initializer=weight_initializer.RandomNormal()),
            MaxPool(size=2, stride=2),
            ConvToFullyConnected(),
            FullyConnected(size=64, activation=activation.tanh),
            FullyConnected(size=10, activation=None, last_layer=True)
        ],
        [
            MaxPool(size=2, stride=2),
            Convolution((16, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh,
                        weight_initializer=weight_initializer.Fill(0), fb_weight_initializer=weight_initializer.RandomNormal(1/np.sqrt(16*16*16))),
            MaxPool(size=2, stride=2),
            Convolution((16, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh,
                        weight_initializer=weight_initializer.Fill(0), fb_weight_initializer=weight_initializer.RandomNormal(1/np.sqrt(8*8*16))),
            MaxPool(size=2, stride=2),
            Convolution((32, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh,
                        weight_initializer=weight_initializer.Fill(0), fb_weight_initializer=weight_initializer.RandomNormal(1/np.sqrt(4*4*32))),
            MaxPool(size=2, stride=2),
            ConvToFullyConnected(),
            FullyConnected(size=64, activation=activation.tanh),
            FullyConnected(size=10, activation=None, last_layer=True)
        ]
    ]

    statistics = []

    for model_layer in model_layers:

        model = Model(
            layers=model_layer,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
            # regularization=0.001,
            # lr_decay=0.5,
            # lr_decay_interval=100
        )

        print("\nRun training:\n------------------------------------")

        stats = model.train(data_set=data, method='dfa', num_passes=5, batch_size=50)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")
        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        statistics.append(stats)

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    labels = []
    for i in range(len(initializers)):
        stats = statistics[i]
        plt.plot(np.arange(len(stats['train_loss'])), stats['train_loss'])
        # plt.plot(stats['valid_step'], stats['valid_loss'])
        labels.append("{}: train loss".format(initializers[i]))
        # labels.append("{}: validation loss".format(initializers[i]))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for i in range(len(initializers)):
        stats = statistics[i]
        plt.plot(np.arange(len(stats['train_accuracy'])), stats['train_accuracy'])
        # plt.plot(stats['valid_step'], stats['valid_accuracy'])
        labels.append("{}: train accuracy".format(initializers[i]))
        # labels.append("{}: validation accuracy".format(initializers[i]))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()