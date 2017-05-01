from dataset.cifar10.loader import Cifar10
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution_im2col import Convolution
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDMomentumOptimizer, GDOptimizer


def simple_conv_3_bp():
    """
    epochs: 10
    lr: 0.001
    reg: 0.0001
    time spend during forward pass: 491.8145480155945
    time spend during backward pass: 664.1201455593109
    time spend in total: 1300.0478966236115
    accuracy: 0.2857
    --------
    epochs: 20
    lr: 0.0001
    reg: 0.0001
    time spend during forward pass: 919.5793607234955
    time spend during backward pass: 1256.7614834308624
    time spend in total: 2426.614329814911
    accuracy: 0.2307
    --------
    epochs: 20
    lr: 0.0001
    reg: 0.001
    time spend during forward pass: 995.856917142868
    time spend during backward pass: 1384.6725234985352
    time spend in total: 2681.251304626465
    accuracy: 0.265
    """
    # Load train data
    cifar = Cifar10('dataset/cifar10')
    X, y = cifar.load_train_set()
    X_test, y_test = cifar.load_test_set()

    layers = [
        Convolution((3, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        Convolution((3, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        Convolution((3, 3, 3, 3), stride=1, padding=1, dropout_rate=0, activation=activation.tanh),
        MaxPool(size=2, stride=2),
        ConvToFullyConnected(),
        FullyConnected(size=10, activation=None, last_layer=True)
    ]

    model = Model(
        layers=layers,
        input_size=(3, 32, 32),
        num_classes=10,
        optimizer=GDMomentumOptimizer(lr=0.001, mu=0.9),
        # optimizer=GDOptimizer(lr=0.001),
        method='bp',
        regularization=0.0001
    )

    model.train(X, y, num_passes=20, batch_size=512)
    model.test(X_test, y_test)

    return model
