import numpy as np

from network.layer import Layer


class Optimizer(object):
    def update(self, layer: Layer, dW: np.ndarray, db: np.ndarray) -> Layer:
        pass


class GDOptimizer(Optimizer):
    def __init__(self, lr: float=0.001) -> None:
        self.lr = lr

    def update(self, layer: Layer, dW: np.ndarray, db: np.ndarray) -> Layer:

        layer.W += -self.lr * dW
        layer.b += -self.lr * db

        return layer
