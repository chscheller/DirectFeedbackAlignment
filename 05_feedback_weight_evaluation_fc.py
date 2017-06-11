from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters
import scipy.interpolate
import dataset.mnist_dataset
import dataset.cifar10_dataset

from network import activation, weight_initializer
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    freeze_support()

    data = dataset.cifar10_dataset.load()

    num_hidden_units = 500
    num_hidden_layers = 5
    num_passes = 30

    initializers = [
        weight_initializer.RandomUniform(-1, 1),
        weight_initializer.RandomUniform(-1/np.sqrt(num_hidden_units), 1/np.sqrt(num_hidden_units)),
        weight_initializer.RandomUniform(-100, 100),
        weight_initializer.RandomNormal(),
        weight_initializer.RandomNormal(1/np.sqrt(num_hidden_units)),
        weight_initializer.RandomNormal(100),
    ]

    labels = [
        'Uniform(low=-1, high=1)',
        'Uniform(low=-1/sqrt(fan_out), high=1/sqrt(fan_out))',
        'Uniform(low=-100, high=100)',
        'Normal(sigma=1, mu=0)',
        'Normal(sigma=1/sqrt(fan_out), mu=0)',
        'Uniform(low=-100, high=100)',
    ]

    statistics = []

    for initializer in initializers:
        layers = [ConvToFullyConnected()]
        for i in range(num_hidden_layers):
            layers += [FullyConnected(size=num_hidden_units, activation=activation.tanh, fb_weight_initializer=initializer)]
        layers += [FullyConnected(size=10, activation=None, last_layer=True)]

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9)
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
