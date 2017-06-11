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
        n, c, h_in, w_in = X.shape
        self.a_in = X
        self.a_in_reshaped = X.reshape(n, c, h_in // self.size, self.size, w_in // self.size, self.size)
        self.a_out = self.a_in_reshaped.max(axis=3).max(axis=4)
        return self.a_out

    def dfa(self, E: np.ndarray) -> tuple:
        return 0, 0

    def back_prob(self, E: np.ndarray) -> tuple:
        dX = (self.a_in_reshaped == self.a_out[:, :, :, np.newaxis, :, np.newaxis]).astype(float)
        dX *= E[:, :, :, np.newaxis, :, np.newaxis]
        dX = dX.reshape(self.a_in.shape)
        return dX, 0, 0

