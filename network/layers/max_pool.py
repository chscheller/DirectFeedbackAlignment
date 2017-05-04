import numpy as np

from network.layer import Layer


class MaxPool(Layer):
    def __init__(self, size, stride):
        super().__init__()
        self.size = size
        self.stride = stride
        self.a_in = None

    def initialize(self, input_size: tuple, num_classes: int, train_method: str) -> tuple:
        assert np.size(input_size) == 3

        c, h_in, w_in = input_size

        assert (h_in - self.size) % self.stride == 0, \
            "pool size ({}) not compatible with input width ({})".format(self.size, h_in)
        assert (w_in - self.size) % self.stride == 0, \
            "pool size ({}) not compatible with input height ({})".format(self.size, h_in)

        self.h_out = ((h_in - self.size) // self.stride) + 1
        self.w_out = ((w_in - self.size) // self.stride) + 1

        return c, self.h_out, self.w_out

    def forward(self, X: np.ndarray, mode='predict') -> np.ndarray:
        self.a_in = X

        n, c, h_in, w_in = X.shape

        out = np.zeros((n, c, self.h_out, self.w_out))
        self.switches = np.zeros((n, c, self.h_out, self.w_out, self.size, self.size))

        h = 0
        for k in range(self.h_out):
            w = 0
            for l in range(self.w_out):
                x_slice = self.a_in[:, :, h:(h + self.size), w:(w + self.size)]
                max_values = np.max(x_slice, axis=(2, 3))
                # noinspection PyUnresolvedReferences
                self.switches[:, :, k, l, :, :] = (x_slice == max_values[:, :, np.newaxis, np.newaxis]).astype(int)
                out[:, :, k, l] = max_values
                w += self.stride
            h += self.stride

        return out

    def dfa(self, E: np.ndarray) -> tuple:
        return 0, 0

    def back_prob(self, E: np.ndarray) -> tuple:
        n, c, h_in, w_in = E.shape
        n, c, h_out, w_out = self.a_in.shape

        dX = np.zeros((n, c, h_out, w_out))

        h = 0
        for k in range(h_in):
            w = 0
            for l in range(w_in):
                dX[:, :, h:(h + self.size), w:(w + self.size)] = np.einsum('ijkl,ij->ijkl', self.switches[:, :,  k, l, :, :], E[:, :, k, l])
                w += self.stride
            h += self.stride
        return dX, 0, 0
