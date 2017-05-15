from dataset.cifar10.loader import Cifar10
from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution_im2col import Convolution
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDOptimizer, GDMomentumOptimizer


def simple_conv_3_dfa():
    """
    epochs: 20
    lr: 0.0001
    reg: 0.0001
    time spend during forward pass: 929.4512648582458
    time spend during backward pass: 559.0829327106476
    time spend in total: 1751.968334197998
    accuracy: 0.2697
    --------
    epochs: 20
    lr: 0.01
    reg: 0.0001
    time spend during forward pass: 1132.449740409851
    time spend during backward pass: 518.5944745540619
    time spend in total: 2037.055358171463
    accuracy: 0.3226
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
        optimizer=GDMomentumOptimizer(lr=0.01, mu=0.9),
        method='dfa',
        regularization=0.0001
    )

    model.train(X, y, num_passes=20, batch_size=256)
    model.test(X_test, y_test)

    return model
