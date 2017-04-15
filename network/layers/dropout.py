import numpy as np

from network.layer import Layer


class Dropout(Layer):

    def __init__(self, rate=0.5) -> None:
        self.rate = rate

    def initialize(self, input_size, out_layer_size, train_method) -> tuple:
        return input_size

    def forward(self, X) -> np.ndarray:
        return X * np.random.binomial(size=X.shape, n=1, p=1 - self.rate)

    def back_prob(self, e, reg, lr) -> np.ndarray:
        return e

