from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters
import scipy.interpolate

import dataset.mnist_dataset
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer
from network.weight_initializer import RandomNormal, RandomUniform

if __name__ == '__main__':
    freeze_support()

    colors = ['#6666ff', '#ff6666', '#66ff66', '#0000ff', '#ff0000', '#00ff00', '#000099', '#990000', '#009900']
    lines = ['-', '-', '-', '--', '--', '--', ':', ':', ':']
    layer_sizes = [
        [800] * 2,
        [400] * 10,
        [240] * 50
    ]
    iterations = [4] * 3

    data = dataset.mnist_dataset.load('dataset/mnist')
    statistics = []
    labels = []

    # Hyperbolic Tangens
    for layer_size, num_passes in zip(layer_sizes, iterations):
        layers = [ConvToFullyConnected()]
        for size in layer_size:
            layers.append(FullyConnected(size=size, activation=activation.tanh))
        layers.append(FullyConnected(size=10, activation=None, last_layer=True))

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-2, mu=0.9)
        )

        print("\nRun training:\n------------------------------------")

        stats = model.train(data_set=data, method='dfa', num_passes=num_passes, batch_size=64)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")
        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        print("\nTrain statisistics:\n------------------------------------")

        print("time spend during forward pass: {}".format(stats['forward_time']))
        print("time spend during backward pass: {}".format(stats['backward_time']))
        print("time spend during update pass: {}".format(stats['update_time']))
        print("time spend in total: {}".format(stats['total_time']))

        labels.append("{}x{} tanh".format(len(layer_size), layer_size[0]))
        statistics.append(stats)

    # Sigmoid
    for layer_size, num_passes in zip(layer_sizes, iterations):
        layers = [ConvToFullyConnected()]
        for size in layer_size:
            layers.append(FullyConnected(
                size=size,
                activation=activation.sigmoid
            ))
        layers.append(FullyConnected(size=10, activation=None, last_layer=True))

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-2, mu=0.9)
        )

        print("\nRun training:\n------------------------------------")

        stats = model.train(data_set=data, method='dfa', num_passes=num_passes, batch_size=64)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")
        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        print("\nTrain statisistics:\n------------------------------------")

        print("time spend during forward pass: {}".format(stats['forward_time']))
        print("time spend during backward pass: {}".format(stats['backward_time']))
        print("time spend during update pass: {}".format(stats['update_time']))
        print("time spend in total: {}".format(stats['total_time']))

        labels.append("{}x{} Sigmoid".format(len(layer_size), layer_size[0]))
        statistics.append(stats)

    for layer_size, num_passes in zip(layer_sizes, iterations):
        layers = [ConvToFullyConnected()]
        layer_count = len(layer_size)
        for size in layer_size:
            layers.append(FullyConnected(
                size=size,
                activation=activation.relu,
                weight_initializer=RandomUniform(low=-np.sqrt(2.0/size), high=np.sqrt(2.0/size)),
                fb_weight_initializer=RandomUniform(low=-np.sqrt(2.0/size), high=np.sqrt(2.0/size))
            ))
        layers.append(FullyConnected(size=10, activation=None, last_layer=True))

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9)
        )

        print("\nRun training:\n------------------------------------")

        stats = model.train(data_set=data, method='dfa', num_passes=num_passes, batch_size=64)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")
        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        print("\nTrain statisistics:\n------------------------------------")

        print("time spend during forward pass: {}".format(stats['forward_time']))
        print("time spend during backward pass: {}".format(stats['backward_time']))
        print("time spend during update pass: {}".format(stats['update_time']))
        print("time spend in total: {}".format(stats['total_time']))

        labels.append("{}x{} leaky ReLU".format(len(layer_size), layer_size[0]))
        statistics.append(stats)

    plt.title('Loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for color, line, stats in zip(colors, lines, statistics):
        train_loss = scipy.ndimage.filters.gaussian_filter1d(stats['train_loss'], sigma=9.5)
        plt.plot(np.arange(len(stats['train_loss'])), train_loss, linestyle=line, color=color)
    plt.legend(labels, loc='best')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for color, line, stats in zip(colors, lines, statistics):
        train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats['train_accuracy'], sigma=9.5)
        plt.plot(np.arange(len(stats['train_accuracy'])), train_accuracy, linestyle=line, color=color)
    plt.legend(labels, loc='lower right')
    plt.grid(True)
    plt.show()
