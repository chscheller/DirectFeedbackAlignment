from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np

from dataset import mnist_dataset
from network import activation, weight_initializer
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    freeze_support()

    num_hidden_units = 240

    initializers = [
        weight_initializer.Fill(1),
        weight_initializer.Fill(100),
        weight_initializer.RandomUniform(-1, 1),
        weight_initializer.RandomUniform(-1/np.sqrt(num_hidden_units), 1/np.sqrt(num_hidden_units)),
        weight_initializer.RandomUniform(-100, 100),
        weight_initializer.RandomNormal(),
        weight_initializer.RandomNormal(1/np.sqrt(num_hidden_units)),
    ]

    statistics = []

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
            optimizer=GDMomentumOptimizer(lr=1e-4, mu=0.9),
            regularization=0.001,
            # lr_decay=0.5,
            # lr_decay_interval=100
        )

        print("\n\n------------------------------------")

        print("Initialize: {}".format(initializer))

        print("\nRun training:\n------------------------------------")

        data = mnist_dataset.load('dataset/mnist')
        stats = model.train(data_set=data, method='dfa', num_passes=10, batch_size=50)
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
        plt.plot(stats['valid_step'], stats['valid_loss'])
        labels.append("{}: train loss".format(initializers[i]))
        labels.append("{}: validation loss".format(initializers[i]))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for i in range(len(initializers)):
        stats = statistics[i]
        plt.plot(np.arange(len(stats['train_accuracy'])), stats['train_accuracy'])
        plt.plot(stats['valid_step'], stats['valid_accuracy'])
        labels.append("{}: train accuracy".format(initializers[i]))
        labels.append("{}: validation accuracy".format(initializers[i]))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()