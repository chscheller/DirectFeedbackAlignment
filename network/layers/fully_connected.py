import numpy as np
import numba as nb

from network import weight_initializer
from network.activation import Activation
from network.layer import Layer

@nb.jit(nopython=True)
def forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.dot(X, W)
    out += b
    return out


class FullyConnected(Layer):
    def __init__(self, size: int, dropout_rate: float=0, batch_norm: bool=False, activation: Activation=None,
                 last_layer=False, weight_initializer=None, fb_weight_initializer=None):
        super().__init__()
        self.size = size
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation = activation
        self.last_layer = last_layer
        self.weight_initializer = weight_initializer
        self.fb_weight_initializer = fb_weight_initializer

    def initialize(self, input_size: int, num_classes: int, train_method: str) -> int:
        assert np.size(input_size) == 1, \
            "invalid input size: scalar required for fully connected layer"

        # initialize weights
        if self.weight_initializer is None:
            self.W = np.random.uniform(low=-1 / np.sqrt(input_size), high=1 / np.sqrt(input_size), size=(input_size, self.size))
        else:
            self.W = self.weight_initializer.init(dim=(input_size, self.size))

        # initialize feedback weights
        if self.fb_weight_initializer is None:
            self.B = np.random.uniform(low=-1, high=1, size=(num_classes, self.size))
        else:
            self.B = self.fb_weight_initializer.init(dim=(num_classes, self.size))

        # initialize bias units
        self.b = np.zeros(self.size)

        return self.size

    def forward(self, X: np.ndarray, mode='predict') -> np.ndarray:
        self.a_in = X
        z = forward(X, self.W, self.b)  # self.a_in.dot(self.W) + self.b
        self.a_out = z if self.activation is None else self.activation.forward(z)
        if mode == 'train' and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(size=self.a_out.shape, n=1, p=1 - self.dropout_rate)
            self.a_out *= self.dropout_mask
        return self.a_out

    def dfa(self, E: np.ndarray) -> tuple:
        E = E if self.last_layer else E.dot(self.B)
        if self.dropout_rate > 0:
            E *= self.dropout_mask
        if self.activation is not None:
            E *= self.activation.backward(self.a_out)
        dW = np.dot(self.a_in.T, E)
        db = np.sum(E, axis=0)
        return dW, db

    def back_prob(self, E: np.ndarray) -> tuple:
        if self.dropout_rate > 0:
            E *= self.dropout_mask
        if self.activation is not None:
            E *= self.activation.backward(self.a_out)
        dX = E.dot(self.W.T)
        dW = np.dot(self.a_in.T, E)
        db = np.sum(E, axis=0)
        return dX, dW, db

    def has_weights(self) -> bool:
        return True

