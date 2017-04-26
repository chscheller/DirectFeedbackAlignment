import numpy as np

from network.layer import Layer


class ConvToFullyConnected(Layer):

    def initialize(self, input_size: tuple, num_classes: int, train_method: str):
        return np.prod(input_size)

    def forward(self, X: np.ndarray, mode='predict') -> np.ndarray:
        self.input_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def dfa(self, e: np.ndarray) -> tuple:
        return 0, 0

    def back_prob(self, e: np.ndarray) -> tuple:
        return e.reshape(self.input_shape), 0, 0


