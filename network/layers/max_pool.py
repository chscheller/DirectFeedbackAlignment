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

        #slice_shape = (self.a_in.shape[0], self.a_in.shape[1], -1)

        h = 0
        for k in range(h_out):
            w = 0
            for l in range(w_out):
                x_slice = self.a_in[:, :, h:(h + self.size), w:(w + self.size)]
                max_values = np.max(x_slice, axis=(2, 3))
                # noinspection PyUnresolvedReferences
                self.switches[:, :, k, l, :, :] = (x_slice == max_values[:, :, np.newaxis, np.newaxis]).astype(int)
                out[:, :, k, l] = max_values
                #x_slice = self.a_in[:, :, h:(h + self.size), w:(w + self.size)].reshape(slice_shape)
                #max_indices = np.argmax(x_slice, axis=2)
                #print("max_indices: {}".format(max_indices))
                #print(max_indices.shape)
                #print(x_slice.shape)
                #print(x_slice[max_indices, :].shape)
                #print(out[:, :, k, l].shape)
                ## h_max, w_max = np.unravel_index(max_indices, (self.size, self.size))
                #max_values = x_slice[np.arange(x_slice.shape[0]), np.arange(x_slice.shape[1]), max_indices]
                #self.switches[:, :, k, l] = max_indices
                ## print("self.switches: {}".format(self.switches))
                ## print("h_max: {}".format(h_max))
                ## print("w_max: {}".format(w_max))
                ## exit()
                out[:, :, k, l] = max_values  #x_slice[max_indices]
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
                #dX[:, :, h:(h + self.size), w:(w + self.size)] = self.switches[:, :, k, l] * \
                #                                                 (E[:, :, k, l])[:, :, np.newaxis, np.newaxis]
                #filter = np.zeros((n, c, self.size*self.size))
                #filter[self.switches[:, :, k, l]] = 1
                #filter[self.switches[:, :, k, l]].reshape(n, c, self.size, self.size)
                #dX[:, :, h:(h + self.size), w:(w + self.size)] = self.switches[:, :,  k, l, :, :] * E[:, :, k, l]
                dX[:, :, h:(h + self.size), w:(w + self.size)] = np.einsum('ijkl,ij->ijkl', self.switches[:, :,  k, l, :, :], E[:, :, k, l])
                w += self.stride
            h += self.stride
        return dX, 0, 0
