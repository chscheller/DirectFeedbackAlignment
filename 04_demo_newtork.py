from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.interpolate
import scipy.ndimage.filters
import threading

import dataset.cifar10_dataset

from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer

if __name__ == '__main__':
    """
    Goal: Compare DFA and BP training performances with respect to validation/test loss, validation/test accuracy and 
    training time on a fully connected NN
    
    Initial learning rate, regularization and learning rate decay parameters were evaluated
    by hand by comparing the training performance on the validation set for various 
    parameter combinations
    """
    freeze_support()

    num_iteration = 10
    data = dataset.cifar10_dataset.load()

    """ DFA Model definition """
    layers_dfa = [
        ConvToFullyConnected(),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model_dfa = Model(
        layers=layers_dfa,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=3*1e-3, mu=0.9),
        regularization=0.09,
        lr_decay=0.5,
        lr_decay_interval=3
    )

    """ BP Model definition """
    layers_bp = [
        ConvToFullyConnected(),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=500, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model_bp = Model(
        layers=layers_bp,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-2, mu=0.9),
        regularization=0.01,
        lr_decay=0.5,
        lr_decay_interval=3
    )

    print("\nRun training:\n------------------------------------")

    class Trainer(object):
        def __init__(self, model, method) -> None:
            super().__init__()
            self.model = model
            self.method = method

        def __call__(self):
            self.model.train(data_set=data, method=self.method, num_passes=num_iteration, batch_size=64)

    # stats_dfa = model_dfa.train(data_set=data, method='dfa', num_passes=num_iteration, batch_size=64)
    # stats_bp = model_bp.train(data_set=data, method='bp', num_passes=num_iteration, batch_size=64)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    def animate(i):
        ax1.clear()
        ax1.plot(np.arange(len(model_dfa.statistics['train_loss'])), model_dfa.statistics['train_loss'])
        ax1.plot(np.arange(len(model_bp.statistics['train_loss'])), model_bp.statistics['train_loss'])

    dfa_train_thread = threading.Thread(target=Trainer(model_dfa, 'dfa'))
    bp_train_thread = threading.Thread(target=Trainer(model_bp, 'bp'))

    dfa_train_thread.start()
    bp_train_thread.start()

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

    dfa_train_thread.join()
    bp_train_thread.join()

    loss, accuracy = model_dfa.cost(*data.test_set())

    print("\nResult DFA:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))

    loss, accuracy = model_bp.cost(*data.test_set())

    print("\nResult BP:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))