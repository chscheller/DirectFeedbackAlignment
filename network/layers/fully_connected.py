import numpy as np

from network.activation import Activation
from network.layer import Layer


class FullyConnected(Layer):
    def __init__(self, size: int, dropout_rate: float=0, activation: Activation=None, last_layer=False):
        self.size = size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.last_layer = last_layer

    def initialize(self, input_size: int, num_classes: int, train_method: str) -> int:
        assert np.size(input_size) == 1, \
            "invalid input size: scalar required for fully connected layer"

        self.b = np.zeros(self.size)

        if train_method == 'dfa':
            self.W = np.zeros((input_size, self.size))
            self.B = np.random.uniform(low=0.0, high=2.0, size=(num_classes, self.size))
            self.B = self.B - np.mean(self.B)
        elif train_method == 'bp':
            self.W = np.random.randn(input_size, self.size) / np.sqrt(input_size)
        else:
            raise "invalid train method '{}'".format(train_method)

        return self.size

    def forward(self, X: np.ndarray, mode='predict') -> np.ndarray:
        self.a_in = X
        z = self.a_in.dot(self.W) + self.b
        self.a_out = z if self.activation is None else self.activation.forward(z)
        if mode == 'train' and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(size=self.a_out.shape, n=1, p=1 - self.dropout_rate)
            self.a_out *= self.dropout_mask
        return self.a_out

    def dfa(self, E: np.ndarray) -> tuple:
        E = E if self.last_layer else E.dot(self.B)
        if self.dropout_rate > 0:
            E *= self.dropout_mask
        if self.activation is None:
            delta = E
        else:
            delta = E * self.activation.backward(self.a_out)
        # delta = e if (self.activation is None or self.last_layer) else (e * self.activation.backward(self.a_out))
        dW = np.dot(self.a_in.T, delta)
        db = np.sum(delta, axis=0)
        return dW, db

    def back_prob(self, E: np.ndarray) -> tuple:
        if self.dropout_rate > 0:
            E *= self.dropout_mask
        if self.activation is None:
            delta = E
        else:
            delta = E * self.activation.backward(self.a_out)
        # delta = e if (self.activation is None or self.last_layer) else (e * self.activation.backward(self.a_out))
        dX = E.dot(self.W.T)
        dW = np.dot(self.a_in.T, delta)
        db = np.sum(delta, axis=0)
        return dX, dW, db

    def has_weights(self) -> bool:
        return True

