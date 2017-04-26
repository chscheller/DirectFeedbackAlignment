import numpy as np

from network.activation import Activation
from network.layer import Layer


def tanh_d(x):
    return 1 - np.power(x, 2)


class Convolution(Layer):
    def __init__(self, filter_shape, stride, padding, dropout_rate: float=0, activation: Activation=None, last_layer=False) -> None:
        assert len(filter_shape) == 4, \
            "invalid filter shape: 4-tuple required, {}-tuple given".format(len(filter_shape))
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.last_layer = last_layer

    def initialize(self, input_size, num_classes, train_method) -> tuple:
        assert np.size(input_size) == 3, \
            "invalid input size: 3-tuple required for convolution layer"

        c_in, h_in, w_in = input_size
        f, c_f, h_f, w_f = self.filter_shape

        assert c_in == c_f, \
            "input channel dimension ({}) not compatible with filter channel dimension ({})".format(c_in, c_f)
        assert (h_in - h_f + 2 * self.padding) % self.stride == 0, \
            "filter width ({}) not compatible with input width ({})".format(h_f, h_in)
        assert (w_in - w_f + 2 * self.padding) % self.stride == 0, \
            "filter height ({}) not compatible with input height ({})".format(h_f, h_in)

        h_out = ((h_in - h_f + 2 * self.padding) // self.stride) + 1
        w_out = ((w_in - w_f + 2 * self.padding) // self.stride) + 1

        self.W = np.zeros(self.filter_shape)
        self.b = np.ones(f)

        if train_method == 'dfa':
            self.B = np.ndarray((num_classes, f, h_out, w_out))
            for i in range(f):
                b = np.random.uniform(low=0.0, high=2.0, size=(num_classes, h_out, w_out))
                self.B[:, i] = b - np.mean(b)
        elif train_method == 'bp':
            for i in range(f):
                self.W[i] = np.random.randn(c_f, h_f, w_f) / np.sqrt(h_f)
        else:
            raise "invalid train method '{}'".format(train_method)

        return f, h_out, w_out

    def forward(self, X, mode='predict') -> np.ndarray:
        n_in, c, h_in, w_in = X.shape
        n_f, c, h_f, w_f = self.W.shape

        h_out = ((h_in - h_f + 2 * self.padding) // self.stride) + 1
        w_out = ((w_in - w_f + 2 * self.padding) // self.stride) + 1

        z = np.zeros((n_in, n_f, h_out, w_out))

        x_padded = np.lib.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              'constant', constant_values=0)

        # for i in range(n_in):  # for each input X
        #    for j in range(n_f):  # for each filter
        #        for h in range(h_out):
        #            for w in range(w_out):
        #                z[i, j, h, w] = np.sum(x_padded[i, :, h * self.stride:h * self.stride + h_f,
        #                                        w * self.stride:w * self.stride + w_f] * self.W[j]) + self.b[j]

        for h in range(h_out):
            for w in range(w_out):
                for j in range(n_f):  # for each filter
                    z[:, j, h, w] = np.sum(
                        x_padded[:, :, h * self.stride:h * self.stride + h_f, w * self.stride:w * self.stride + w_f] *
                        self.W[j], axis=(1, 2, 3)) + self.b[j]

        self.a_in = X
        self.a_out = z if self.activation is None else self.activation.forward(z)
        if mode == 'train' and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(size=self.a_out.shape, n=1, p=1 - self.dropout_rate)
            self.a_out *= self.dropout_mask
        return self.a_out

    def dfa(self, E: np.ndarray) -> tuple:
        E = np.einsum('ij,jklm->iklm', E, self.B)

        if self.dropout_rate > 0:
            E *= self.dropout_mask

        n_f, c_f, h_f, w_f = self.W.shape
        n_e, c_e, h_e, w_e = E.shape

        delta = E * (self.a_out if self. activation is None else self.activation.backward(self.a_out))
        X_padded = np.lib.pad(self.a_in, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              'constant', constant_values=0)

        dW = np.zeros(self.W.shape)
        db = np.sum(E, axis=(0, 2, 3))

        for h in range(h_e):
            for w in range(w_e):
                dW += np.tensordot(delta[:, :, h, w].T, X_padded[:, :, h * self.stride:h * self.stride + h_f,
                                                        w * self.stride:w * self.stride + w_f], axes=([-1], [0]))

        return dW, db

    def back_prob(self, E: np.ndarray) -> tuple:

        if self.dropout_rate > 0:
            E *= self.dropout_mask

        n_f, c_f, h_f, w_f = self.W.shape
        n_e, c_e, h_e, w_e = E.shape

        dX = np.zeros(self.a_in.shape)
        dW = np.zeros(self.W.shape)

        delta = E * (self.a_out if self. activation is None else self.activation.backward(self.a_out))
        X_padded = np.lib.pad(self.a_in, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              'constant', constant_values=0)
        dX_padded = np.lib.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                               'constant', constant_values=0)

        # for i in range(n_x):
        #    for h in range(h_e):
        #        for w in range(w_e):
        #            for j in range(n_f):
        #                dW[j] += delta[i, j, h, w] * X_padded[i, :, h*self.stride:h*self.stride+h_f, w*self.stride:w*self.stride+w_f]
        #            # print("dX shape: {}".format(dX[i, :, h*self.stride:h*self.stride+h_f, w*self.stride:w*self.stride+w_f].shape))
        #            # print("W shape: {}".format(self.W[j].shape))
        #            # print("delta shape: {}".format(delta[i, j, h, w].shape))
        #            dX_padded[i, :, h*self.stride:h*self.stride+h_f, w*self.stride:w*self.stride+w_f] += self.W[j] *\
        #                                                                                              delta[i, j, h, w]

        for h in range(h_e):
            for w in range(w_e):
                curr_h = h * self.stride
                curr_w = w * self.stride
                dW += np.tensordot(
                    delta[:, :, h, w].T,
                    X_padded[:, :, curr_h:curr_h + h_f, curr_w:curr_w + w_f],
                    axes=([-1], [0])
                )
                dX_padded[:, :, curr_h:curr_h + h_f, curr_w:curr_w + w_f] += np.einsum(
                    'ij,jklm->iklm',
                    delta[:, :, h, w],
                    self.W
                )

        dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        db = np.sum(E, axis=(0, 2, 3))

        return dX, dW, db

    def has_weights(self) -> bool:
        return True

