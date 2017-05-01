from network import activation
from network.layers.conv_to_fully_connected import ConvToFullyConnected
from network.layers.convolution_im2col import Convolution
from network.layers.fully_connected import FullyConnected
from network.layers.max_pool import MaxPool
from network.model import Model
from network.optimizer import GDOptimizer, GDMomentumOptimizer


def simple_conv_3_bp():

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
        method='bp',
        regularization=0.0001
    )

    return model
