import numpy as np

from network.layer import Layer


class MaxPool(Layer):
    def __init__(self, size, stride):
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

        h_out = ((h_in - self.size) // self.stride) + 1
        w_out = ((w_in - self.size) // self.stride) + 1

        return c, h_out, w_out

    def forward(self, X: np.ndarray, mode='predict') -> np.ndarray:
        self.a_in = X

        n, c, h_in, w_in = X.shape

        h_out = ((h_in - self.size) // self.stride) + 1
        w_out = ((w_in - self.size) // self.stride) + 1

        out = np.zeros((n, c, h_out, w_out))
        self.switches = np.zeros((n, c, h_out, w_out, self.size, self.size))

        h = 0
        for k in range(h_out):
            w = 0
            for l in range(w_out):
                x_slice = self.a_in[:, :, h:(h + self.size), w:(w + self.size)]
                max_indices = np.argmax(x_slice.reshape((x_slice.shape[0], x_slice.shape[1], -1)), axis=2)
                h_max, w_max = np.unravel_index(max_indices, (x_slice.shape[2], x_slice.shape[3]))
                self.switches[:, :, k, l, h_max, w_max] = 1
                print("h_max: {}".format(h_max))
                print("w_max: {}".format(w_max))
                exit()
                out[:, :, k, l] = x_slice[:, :, h_max, w_max]
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
                # dX[:, :, self.switches[:, :, k, l, 0], self.switches[:, :, k, l, 1]] = (E[:, :, k, l])[:, :, np.newaxis, np.newaxis]
                dX[:, :, h:(h + self.size), w:(w + self.size)] = self.switches[:, :, k, l] * \
                                                                 (E[:, :, k, l])[:, :, np.newaxis, np.newaxis]
                w += self.stride
            h += self.stride
        return dX, 0, 0
