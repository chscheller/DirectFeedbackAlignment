import numpy as np

from network.activation import Activation
from network.layer import Layer
from network.utils.im2col_cython import im2col_cython, col2im_cython


class Convolution(Layer):
    def __init__(self, filter_shape, stride, padding, dropout_rate: float = 0, batch_norm: bool=False, activation: Activation = None,
                 last_layer=False) -> None:
        assert len(filter_shape) == 4, \
            "invalid filter shape: 4-tuple required, {}-tuple given".format(len(filter_shape))
        super().__init__()
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
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
        self.b = np.zeros(f)

        if train_method == 'dfa':
            self.B = np.ndarray((num_classes, f, h_out, w_out))
            for i in range(f):
                b = np.random.uniform(low=0.0, high=2.0, size=(num_classes, h_out, w_out))
                self.B[:, i] = b - np.mean(b)
        elif train_method == 'bp':
            for i in range(f):
                self.W[i] = np.random.randn(c_f, h_f, w_f) / np.sqrt(c_f * h_f * w_f)
        else:
            raise "invalid train method '{}'".format(train_method)

        return f, h_out, w_out

    def forward(self, X, mode='predict') -> np.ndarray:
        n_in, c, h_in, w_in = X.shape
        n_f, c, h_f, w_f = self.W.shape

        h_out = ((h_in - h_f + 2 * self.padding) // self.stride) + 1
        w_out = ((w_in - w_f + 2 * self.padding) // self.stride) + 1
        # z = np.zeros((n_in, n_f, h_out, w_out), dtype=X.dtype)

        self.x_cols = im2col_cython(X, self.W.shape[2], self.W.shape[3], self.padding, self.stride)
        z = self.W.reshape((self.W.shape[0], -1)).dot(self.x_cols) + self.b.reshape(-1, 1)
        z = z.reshape(self.W.shape[0], h_out, w_out, X.shape[0]).transpose(3, 0, 1, 2)

        self.a_in = X
        self.a_out = z if self.activation is None else self.activation.forward(z)

        if mode == 'train' and self.dropout_rate > 0:
            # self.dropout_mask = np.random.binomial(size=self.a_out.shape, n=1, p=1 - self.dropout_rate)
            self.dropout_mask = (np.random.rand(*self.a_out.shape) > self.dropout_rate).astype(int)
            self.a_out *= self.dropout_mask

        return self.a_out

    def dfa(self, E: np.ndarray) -> tuple:
        E = np.einsum('ij,jklm->iklm', E, self.B)

        if self.dropout_rate > 0:
            E *= self.dropout_mask

        n_f, c_f, h_f, w_f = self.W.shape

        delta = E * (self.a_out if self. activation is None else self.activation.backward(self.a_out))

        delta_reshaped = delta.transpose((1, 2, 3, 0)).reshape(n_f, -1)
        dW = delta_reshaped.dot(self.x_cols.T).reshape(self.W.shape)

        db = np.sum(E, axis=(0, 2, 3))

        return dW, db

    def back_prob(self, E: np.ndarray) -> tuple:
        if self.dropout_rate > 0:
            E *= self.dropout_mask

        n_f, c_f, h_f, w_f = self.W.shape

        delta = E * (self.a_out if self. activation is None else self.activation.backward(self.a_out))

        delta_reshaped = delta.transpose((1, 2, 3, 0)).reshape(n_f, -1)
        dW = delta_reshaped.dot(self.x_cols.T).reshape(self.W.shape)

        dX_cols = self.W.reshape(n_f, -1).T.dot(delta_reshaped)
        dX = col2im_cython(dX_cols, self.a_in.shape[0], self.a_in.shape[1], self.a_in.shape[2], self.a_in.shape[3],
                           h_f, w_f, self.padding, self.stride)

        db = np.sum(E, axis=(0, 2, 3))

        return dX, dW, db

    def has_weights(self) -> bool:
        return True

