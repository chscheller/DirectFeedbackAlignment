import numpy as np

from network.activation import Activation
from network.layer import Layer
from network.utils.im2col_cython import im2col_cython, col2im_cython


class Convolution(Layer):
    def __init__(self, filter_shape, stride, padding, dropout_rate: float = 0, activation: Activation = None,
                 last_layer=False, weight_initializer=None, fb_weight_initializer=None) -> None:
        assert len(filter_shape) == 4, \
            "invalid filter shape: 4-tuple required, {}-tuple given".format(len(filter_shape))
        super().__init__()
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.last_layer = last_layer
        self.weight_initializer = weight_initializer
        self.fb_weight_initializer = fb_weight_initializer

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

        self.h_out = ((h_in - h_f + 2 * self.padding) // self.stride) + 1
        self.w_out = ((w_in - w_f + 2 * self.padding) // self.stride) + 1

        # initialize weights
        if self.weight_initializer is None:
            sqrt_fan_in = np.sqrt(c_in * h_in * w_in)
            self.W = np.random.uniform(low=-1 / sqrt_fan_in, high=1 / sqrt_fan_in, size=self.filter_shape)
        else:
            self.W = self.weight_initializer.init(dim=(f, c_f, h_f, w_f))

        # initialize feedback weights
        if self.fb_weight_initializer is None:
            sqrt_fan_out = np.sqrt(f * self.h_out * self.w_out)
            # self.B = np.random.uniform(low=-1 / sqrt_fan_out, high=1 / sqrt_fan_out, size=(num_classes, f, self.h_out, self.w_out))
            self.B = np.random.uniform(low=-1 / sqrt_fan_out, high=1 / sqrt_fan_out, size=(num_classes, f * self.h_out * self.w_out))
        else:
            # self.B = self.fb_weight_initializer.init(dim=(num_classes, f, self.h_out, self.w_out))
            self.B = self.fb_weight_initializer.init(dim=(num_classes, f * self.h_out * self.w_out))

        # initialize bias units
        self.b = np.zeros(f)

        return f, self.h_out, self.w_out

    def forward(self, X, mode='predict') -> np.ndarray:
        n_in, c, h_in, w_in = X.shape
        n_f, c, h_f, w_f = self.W.shape

        self.x_cols = im2col_cython(X, h_f, w_f, self.padding, self.stride)  # <->
        z = self.W.reshape((n_f, -1)).dot(self.x_cols)
        z += self.b.reshape(-1, 1)  # +
        z = z.reshape(n_f, self.h_out, self.w_out, n_in).transpose(3, 0, 1, 2)

        self.a_in = X

        if self.activation is None:
            self.a_out = z
        else:
            self.a_out = self.activation.forward(z)

        if mode == 'train' and self.dropout_rate > 0:
            # self.dropout_mask = np.random.binomial(size=self.a_out.shape, n=1, p=1 - self.dropout_rate)
            self.dropout_mask = (np.random.rand(*self.a_out.shape) > self.dropout_rate).astype(int)
            self.a_out *= self.dropout_mask

        return self.a_out

    def dfa(self, E: np.ndarray) -> tuple:
        # E = np.einsum('ij,jklm->iklm', E, self.B)

        n_f, c_f, h_f, w_f = self.W.shape

        E = np.dot(E, self.B).reshape((-1, n_f, self.h_out, self.w_out))
        if self.dropout_rate > 0:
            E *= self.dropout_mask

        if self.activation is None:
            E *= self.a_out
        else:
            E *= self.activation.gradient(self.a_out)

        dW = E.transpose((1, 2, 3, 0)).reshape(n_f, -1).dot(self.x_cols.T).reshape(self.W.shape)
        db = np.sum(E, axis=(0, 2, 3))

        return dW, db

    def back_prob(self, E: np.ndarray) -> tuple:
        if self.dropout_rate > 0:
            E *= self.dropout_mask

        n_in, c_in, h_in, w_in = self.a_in.shape
        n_f, c_f, h_f, w_f = self.W.shape

        if self.activation is None:
            E *= self.a_out
        else:
            E *= self.activation.gradient(self.a_out)
        delta_reshaped = E.transpose((1, 2, 3, 0)).reshape(n_f, -1)

        dX_cols = self.W.reshape(n_f, -1).T.dot(delta_reshaped)
        dX = col2im_cython(dX_cols, n_in, c_in, h_in, w_in, h_f, w_f, self.padding, self.stride)
        dW = delta_reshaped.dot(self.x_cols.T).reshape(self.W.shape)
        db = np.sum(E, axis=(0, 2, 3))

        return dX, dW, db

    def has_weights(self) -> bool:
        return True
