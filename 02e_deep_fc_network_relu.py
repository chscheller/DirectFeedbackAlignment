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

    colors = ['blue', 'red', 'green', 'black']
    depths = [10]
    iterations = [4]

    data = dataset.mnist_dataset.load('dataset/mnist')
    statistics = []

    for depth, num_passes in zip(depths, iterations):
        layers = [ConvToFullyConnected()] + \
                 [FullyConnected(size=240, activation=activation.leaky_relu,
                                 #weight_initializer=RandomNormal(sigma=np.sqrt(2.0/240)),
                                 weight_initializer=RandomUniform(low=-np.sqrt(1.0/240), high=np.sqrt(1.0/240)),
                                 fb_weight_initializer=RandomUniform(low=-np.sqrt(1.0/240), high=np.sqrt(1.0/240))) for _ in range(depth)] + \
                 [FullyConnected(size=10, activation=None, last_layer=True)]

        """ BP """

        model = Model(
            layers=layers,
            num_classes=10,
            optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9)
        )

        print("\nRun training:\n------------------------------------")

        stats_bp = model.train(data_set=data, method='dfa', num_passes=num_passes, batch_size=64)
        loss, accuracy = model.cost(*data.test_set())

        print("\nResult:\n------------------------------------")
        print('loss on test set: {}'.format(loss))
        print('accuracy on test set: {}'.format(accuracy))

        print("\nTrain statisistics:\n------------------------------------")

        print("time spend during forward pass: {}".format(stats_bp['forward_time']))
        print("time spend during backward pass: {}".format(stats_bp['backward_time']))
        print("time spend during update pass: {}".format(stats_bp['update_time']))
        print("time spend in total: {}".format(stats_bp['total_time']))

        statistics.append(stats_bp)

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for color, stats in zip(colors, statistics):
        train_loss = scipy.ndimage.filters.gaussian_filter1d(stats['train_loss'], sigma=10)
        plt.plot(np.arange(len(stats['train_loss'])), train_loss, linestyle='-', color=color)
    legends = []
    for depth in depths:
        legends.append('{}xfc leaky relu, train loss bp'.format(depth))
    plt.legend(legends, loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for color, stats in zip(colors, statistics):
        train_accuracy = scipy.ndimage.filters.gaussian_filter1d(stats['train_accuracy'], sigma=10)
        plt.plot(np.arange(len(stats['train_accuracy'])), train_accuracy, linestyle='-', color=color)
    legends = []
    for depth in depths:
        legends.append('{}xfc leaky relu, train accuracy bp'.format(depth))
    plt.legend(legends, loc='lower right')
    plt.grid(True)
    plt.show()
