import numpy as np

from network.layer import Layer


class Dropout(Layer):

    def __init__(self, rate=0.5) -> None:
        self.rate = rate

    def initialize(self, input_size, out_layer_size, train_method) -> tuple:
        return input_size

    def forward(self, X, mode='predict') -> np.ndarray:
        if mode == 'train':
            self.dropout_mask = np.random.binomial(size=X.shape, n=1, p=1 - self.rate)
            return X * self.dropout_mask
        else:
            return X

    def dfa(self, E: np.ndarray) -> tuple:
        return 0, 0

    def back_prob(self, E: np.ndarray) -> tuple:
        return E * self.dropout_mask, 0, 0



