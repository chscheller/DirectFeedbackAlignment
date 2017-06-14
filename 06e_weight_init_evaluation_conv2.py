from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters
import scipy.interpolate
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

    data = dataset.cifar10_dataset.load()

    num_passes = 30

    initializers = [[], [], [], [], [], [], []]

    for i in [8*16*16, 16*8*8, 32*4*4]:
        initializers[0].append(weight_initializer.Fill(0)),
        initializers[1].append(weight_initializer.Fill(1e-3)),
        initializers[2].append(weight_initializer.Fill(1)),
        initializers[3].append(weight_initializer.RandomUniform(-1, 1))
        initializers[4].append(weight_initializer.RandomUniform(-1/np.sqrt(i), 1/np.sqrt( i)))
        initializers[5].append(weight_initializer.RandomNormal())
        initializers[6].append(weight_initializer.RandomNormal(1/np.sqrt(i)))

    labels = [
        'Fill(0)',
        'Fill(0.001)',
        'Fill(1)',
        'Uniform(low=-1, high=1)',
        'Uniform(low=-1/sqrt(fan_out), high=1/sqrt(fan_out))',
        'Normal(sigma=1, mu=0)',
        'Normal(sigma=1/sqrt(fan_out), mu=0)',
    ]

    statistics = []
    for initializer in initializers:
        layers = [
            MaxPool(size=2, stride=2),
            Convolution((8, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh, weight_initializer=initializer[0]),
            MaxPool(size=2, stride=2),
            Convolution((16, 8, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh, weight_initializer=initializer[1]),
            MaxPool(size=2, stride=2),
            Convolution((32, 16, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh, weight_initializer=initializer[2]),
            MaxPool(size=2, stride=2),
            ConvToFullyConnected(),
            FullyConnected(size=64, activation=activation.tanh),
            FullyConnected(size=10, activation=None, last_layer=True)
        ]

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
        )

        print("\n\n------------------------------------")

        print("Initialize: {}".format(initializer))

        print("\nRun training:\n------------------------------------")

        stats = model.train(data_set=data, method='dfa', num_passes=num_passes, batch_size=50)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")
        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        statistics.append(stats)

    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for stats in statistics:
        train_loss = scipy.ndimage.filters.gaussian_filter1d(stats['train_loss'], sigma=10)
        plt.plot(np.arange(len(stats['train_loss'])), train_loss)
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for stats in statistics:
        train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats['train_accuracy'], sigma=10)
        plt.plot(np.arange(len(stats['train_accuracy'])), train_accuracy)
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()