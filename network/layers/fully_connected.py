import numpy as np

from network.layer import Layer


def tanh_d(x):
    return 1 - np.power(x, 2)


class FullyConnected(Layer):
    def __init__(self, size, activation=np.tanh, activation_d=tanh_d, last_layer=False):
        """
        :type activation_d: Any
        """
        self.size = size
        self.activation = activation
        self.activation_d = activation_d
        self.last_layer = last_layer

    def initialize(self, input_size, out_layer_size, train_method):
        assert np.size(input_size) == 1

        self.b = np.zeros(self.size)

        if train_method == 'dfa':
            self.W = np.zeros((input_size, self.size))
            self.B = np.random.uniform(low=0.0, high=2.0, size=(out_layer_size, self.size))
            self.B = self.B - np.mean(self.B)
        elif train_method == 'bp':
            self.W = np.random.randn(input_size, self.size) / np.sqrt(input_size)
        else:
            raise "invalid train method '{}'".format(train_method)

        return self.size

    def forward(self, X):
        self.a_in = X
        z = self.a_in.dot(self.W) + self.b
        self.a_out = self.activation(z)
        return self.a_out

    def dfa(self, e, reg, lr):
        delta = e if self.last_layer else (e.dot(self.B) * self.activation_d(self.a_out))
        self.__update_weights(delta, reg, lr)
        return self

    def back_prob(self, e, reg, lr):
        delta = e if self.last_layer else (e * self.activation_d(self.a_out))
        self.__update_weights(delta, reg, lr)
        return e.dot(self.W.T)

    def __update_weights(self, delta, reg, lr):
        dW = np.dot(self.a_in.T, delta) + reg * self.W
        db = np.sum(delta, axis=0)
        self.W += -lr * dW
        self.b += -lr * db

    def sum_weights(self):
        return np.sum(np.square(self.W))
