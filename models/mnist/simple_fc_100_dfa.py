import matplotlib.pyplot as plt
import numpy as np

from dataset import mnist_dataset
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.fully_connected import FullyConnected
from network.model import Model
from network.optimizer import GDMomentumOptimizer

def simple_fc_100_dfa():

    data = mnist_dataset.load('dataset/mnist')

    layers = [
        ConvToFullyConnected(),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),

        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),

        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=240, activation=activation.tanh),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model = Model(
        layers=layers,
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=1e-4, mu=0.9),
        regularization=0.00,
        # lr_decay=0.5,
        # lr_decay_interval=100
    )

    print("\nRun training:\n------------------------------------")

    stats = model.train(data_set=data, method='dfa', num_passes=10, batch_size=50)
    loss, accuracy = model.cost(*data.test_set())

    print("\nResult:\n------------------------------------")
    print('loss on test set: {}'.format(loss))
    print('accuracy on test set: {}'.format(accuracy))

    print("\nTrain statisistics:\n------------------------------------")

    print("time spend during forward pass: {}".format(stats['forward_time']))
    print("time spend during backward pass: {}".format(stats['backward_time']))
    print("time spend during update pass: {}".format(stats['update_time']))
    print("time spend in total: {}".format(stats['total_time']))

    plt.title('Loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(len(stats['train_loss'])), stats['train_loss'])
    plt.plot(stats['valid_step'], stats['valid_loss'])
    plt.legend(['train loss', 'validation loss'], loc='upper right')
    plt.grid(True)
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(np.arange(len(stats['train_accuracy'])), stats['train_accuracy'])
    plt.plot(stats['valid_step'], stats['valid_accuracy'])
    plt.legend(['train accuracy', 'validation accuracy'], loc='upper right')
    plt.grid(True)
    plt.show()

    return model
