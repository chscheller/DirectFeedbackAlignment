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

if __name__ == '__main__':
    freeze_support()

    # colors = [('deepskyblue', 'darkblue'), ('red', 'maroon'), ('goldenrod', 'sienna'), ('limegreen', 'darkgreen'),
    #           ('purple', 'magenta'), ('gray', 'black')]
    # depths = [10, 15, 20, 30, 50, 100]
    # iterations = ([2] * 5) + [4]

    colors = [('blue', 'blue'), ('red', 'red'), ('green', 'green'), ('orange', 'orange'), ('black', 'black')]
    depths = [10, 15, 20, 50, 100]
    iterations = ([4] * 5)

    # colors = [('blue', 'blue')]
    # depths = [20]
    # iterations = [20]

    data = dataset.mnist_dataset.load('dataset/mnist')
    statistics = []

    for depth, num_passes in zip(depths, iterations):
        layers = [ConvToFullyConnected()] + \
                 [FullyConnected(size=240, activation=activation.tanh) for _ in range(depth)] + \
                 [FullyConnected(size=10, activation=None, last_layer=True)]

        """ DFA """

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
        )

        print("\nRun training:\n------------------------------------")

        stats_dfa = model.train(data_set=data, method='dfa', num_passes=num_passes, batch_size=64)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")

        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        print("\nTrain statisistics:\n------------------------------------")

        print("time spend during forward pass: {}".format(stats_dfa['forward_time']))
        print("time spend during backward pass: {}".format(stats_dfa['backward_time']))
        print("time spend during update pass: {}".format(stats_dfa['update_time']))
        print("time spend in total: {}".format(stats_dfa['total_time']))

        """ BP """

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-2, mu=0.9)
        )

        print("\nRun training:\n------------------------------------")

        stats_bp = model.train(data_set=data, method='bp', num_passes=num_passes, batch_size=64)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")
        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        print("\nTrain statisistics:\n------------------------------------")

        print("time spend during forward pass: {}".format(stats_bp['forward_time']))
        print("time spend during backward pass: {}".format(stats_bp['backward_time']))
        print("time spend during update pass: {}".format(stats_bp['update_time']))
        print("time spend in total: {}".format(stats_bp['total_time']))

        statistics.append((stats_dfa, stats_bp))

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for color, stats in zip(colors, statistics):
        color_dfa = color[0]
        stats_dfa = stats[0]
        color_bp = color[1]
        stats_bp = stats[1]
        dfa_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_loss'], sigma=10)
        bp_train_loss = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_loss'], sigma=10)
        plt.plot(np.arange(len(stats_dfa['train_loss'])), dfa_train_loss, color=color_dfa)
        plt.plot(np.arange(len(stats_bp['train_loss'])), bp_train_loss, linestyle='--', color=color_bp)
    legends = []
    for depth in depths:
        legends.append('{}xfc tanh, train loss dfa'.format(depth))
        legends.append('{}xfc tanh, train loss bp'.format(depth))
    plt.legend(legends, loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for color, stats in zip(colors, statistics):
        color_dfa = color[0]
        stats_dfa = stats[0]
        color_bp = color[1]
        stats_bp = stats[1]
        dfa_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_dfa['train_accuracy'], sigma=10)
        bp_train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats_bp['train_accuracy'], sigma=10)
        plt.plot(np.arange(len(stats_dfa['train_accuracy'])), dfa_train_accuracy, color=color_dfa)
        plt.plot(np.arange(len(stats_bp['train_accuracy'])), bp_train_accuracy, linestyle='--', color=color_bp)
    legends = []
    for depth in depths:
        legends.append('{}xfc tanh, train accuracy dfa'.format(depth))
        legends.append('{}xfc tanh, train accuracy bp'.format(depth))
    plt.legend(legends, loc='lower right')
    plt.grid(True)
    plt.show()
