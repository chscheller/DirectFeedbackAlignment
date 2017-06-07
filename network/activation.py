import numpy as np
from numba import jitclass


class Activation(object):
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def backward(self, x: np.ndarray) -> np.ndarray:
        pass


@jitclass([])
class __TanH(Activation):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        # noinspection PyTypeChecker
        return 1 - np.power(x, 2)


@jitclass([])
class __Softmax(Activation):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        # TODO
        raise Exception("Not yet implemented!")


@jitclass([])
class __Sigmoid(Activation):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)


@jitclass([])
class __ReLU(Activation):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x > 0


@jitclass([])
class __LeakyReLU(Activation):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.01 * x, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return 0.01 + 0.99 * (x > 0)


tanh = __TanH()
softmax = __Softmax()
sigmoid = __Sigmoid()
relu = __ReLU()
leaky_relu = __LeakyReLU()
