import numpy as np

from network.layer import Layer


class ConvToFullyConnected(Layer):

    def initialize(self, input_size, out_layer_size, train_method):
        return np.prod(input_size)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def back_prob(self, e, reg, lr):
        return e.reshape(self.input_shape)
