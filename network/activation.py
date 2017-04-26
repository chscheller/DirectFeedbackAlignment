import numpy as np


class Activation(object):
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def backward(self, x: np.ndarray) -> np.ndarray:
        pass


class __TanH(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        # noinspection PyTypeChecker
        return 1 - np.power(x, 2)


class __Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        # TODO
        raise Exception("Not yet implemented!")


tanh = __TanH()
softmax = __Softmax()
