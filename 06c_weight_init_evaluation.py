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
    """
    Evaluate which Initialization strategy leads to better performance:
    
        - Normal(1/sqrt(fan_in), 0)
        - Normal(1/sqrt(fan_out), 0)
    
    Both strategies are evaluated on how they affect loss/accuracy when 
    training a network with dfa and backpropagation
    """

    freeze_support()

    data = dataset.mnist_dataset.load('dataset/mnist')
    fan_in = [784, 200, 400, 400, 400]
    fan_out = [200, 400, 400, 400, 200]

    # data = dataset.cifar10_dataset.load()
    # fan_in = [3072, 200, 400, 400, 400]
    # fan_out = [200, 400, 400, 400, 200]

    initializers = ['Normal(1/sqrt(fan_in), 0)', 'Normal(1/sqrt(fan_out), 0)']
    train_methods = ['dfa', 'bp']

    statistics = []
    labels = []

    for sizes, initializer in zip([fan_in, fan_out], initializers):
        for train_method in train_methods:
            layers = [ConvToFullyConnected()]

            for i in range(len(sizes)):
                layers.append(FullyConnected(
                    size=fan_out[i],
                    activation=activation.tanh,
                    weight_initializer=weight_initializer.RandomNormal(1/np.sqrt(sizes[i]))
                )),

            layers.append(FullyConnected(size=10, activation=None, last_layer=True))

            model = Model(
                layers=layers,
                num_classes=10,
                optimizer=GDMomentumOptimizer(lr=1e-3, mu=0.9),
                regularization=0.001,
                # lr_decay=0.5,
                # lr_decay_interval=100
            )

            print("\nRun training:\n------------------------------------")

            stats = model.train(data_set=data, method=train_method, num_passes=3, batch_size=50)
            loss, accuracy = model.cost(*data.test_set())

            print("\nResult:\n------------------------------------")
            print('loss on test set: {}'.format(loss))
            print('accuracy on test set: {}'.format(accuracy))

            statistics.append(stats)
            labels.append('{}, {}'.format(train_method, initializer))

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    legends = []
    for stats, label in zip(statistics, labels):
        plt.plot(np.arange(len(stats['train_loss'])), stats['train_loss'])
        legends.append("{}: train loss".format(label))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    for stats, label in zip(statistics, labels):
        plt.plot(np.arange(len(stats['train_accuracy'])), stats['train_accuracy'])
        legends.append("{}: train accuracy".format(label))
    plt.legend(labels, loc='upper right')
    plt.grid(True)
    plt.show()