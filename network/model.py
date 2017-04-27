import multiprocessing
from typing import Iterable

import numpy as np

from network.helpers import data
from network.layer import Layer
from network.layers.convolution import Convolution
from network.loss import SoftmaxCrossEntropyLoss, Loss
from network.optimizer import GDOptimizer, Optimizer


class DFA(object):
    def __init__(self, e: np.ndarray) -> None:
        self.e = e

    def __call__(self, layer: Layer) -> tuple:
        return (layer,) + layer.dfa(self.e)


class UpdateLayer(object):
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def __call__(self, gradients: np.ndarray) -> Layer:
        layer, dW, db = gradients
        if layer.has_weights():
            return self.optimizer.update(layer, dW, db)
        else:
            return layer


class Model(object):
    def __init__(
            self,
            layers: Iterable[Layer],
            num_classes: int,
            loss: Loss=SoftmaxCrossEntropyLoss(),
            optimizer: Optimizer=GDOptimizer(),
            regularization: float=0
    ) -> None:
        self.layers = layers
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.regularization = regularization
        self.loss = loss

    def cost(self, X, y):
        n = X.shape[0]

        """ forward pass """
        for layer in self.layers:
            X = layer.forward(X)

        """ loss """
        loss, _ = self.loss.calculate(X, y)

        """ regularization (L2) """
        if self.regularization > 0:
            total_weights = 0
            for layer in self.layers:
                if layer.has_weights():
                    total_weights += np.sum(np.square(layer.W))
            loss += (total_weights * self.regularization / 2.) / n

        return loss

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X, mode='train')
        return np.argmax(X, axis=1)

    def gradient_check(self, x, y, method='dfa'):
        x_in = x

        """ initalize layers """
        input_size = x.shape[1:]
        for layer in self.layers:
            input_size = layer.initialize(input_size, self.num_classes, method)

        """ forward pass """
        for layer in self.layers:
            x = layer.forward(x, mode='train')

        """ loss """
        loss, delta = self.loss.calculate(x, y)

        def calc_cost(xx):
            for l in self.layers:
                xx = l.forward(xx, mode='train')
            c, _ = self.loss.calculate(xx, y)
            return c

        """ backward pass """
        gradients = []
        if method == 'dfa':
            gradients = map(DFA(delta), self.layers)
        else:
            dX = delta
            epsilon = 0.0001
            for layer in reversed(self.layers):
                dX, dW, _ = layer.back_prob(dX)
                if layer.has_weights():
                    W_orig = layer.W.copy()
                    W_unrolled = np.reshape(layer.W, -1)
                    dW_unrolled = dW.reshape(-1)
                    dW_approx = np.zeros(dW_unrolled.shape)
                    for i in range(dW_unrolled.size):
                        wPlus = W_unrolled.copy()
                        wPlus[i] += epsilon
                        layer.W = wPlus.reshape(W_orig.shape)
                        costPlus = self.cost(x_in, y)
                        wMinus = W_unrolled.copy()
                        wMinus[i] -= epsilon
                        layer.W = wMinus.reshape(W_orig.shape)
                        costMinus = self.cost(x_in, y)
                        dW_approx[i] = (costPlus - costMinus) / (2. * epsilon)
                    layer.W = W_orig
                    print('calculated: {}', dW_unrolled)
                    print('approx: {}', dW_approx)
                    print("relative difference {}".format(np.linalg.norm(dW_approx - dW_unrolled)/np.linalg.norm(dW_approx + dW_unrolled)))


    def train(self, X, y, num_passes=2000, batch_size=128, method='dfa'):
        y_train, X_train, y_valid, X_valid = data.split_train_valid_data(X, y)

        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        """ initalize layers """
        input_size = X.shape[1:]
        for layer in self.layers:
            input_size = layer.initialize(input_size, self.num_classes, method)

        for epoch in range(num_passes):
            batch_no = 0
            for batch in data.slice_mini_batches(X_train, y_train, batch_size):
                X_batch, y_batch = batch

                """ forward pass """
                for layer in self.layers:
                    X_batch = layer.forward(X_batch, mode='train')

                """ loss """
                loss, delta = self.loss.calculate(X_batch, y_batch)

                """ backward pass """
                gradients = []
                if method == 'dfa':
                    gradients = pool.map(DFA(delta), self.layers)
                else:
                    dX = delta
                    for layer in reversed(self.layers):
                        dX, dW, db = layer.back_prob(dX)
                        gradients.append((layer, dW, db))
                    gradients.reverse()

                """ regularization (L2) """
                if self.regularization > 0:
                    for layer, dW, db in gradients:
                        if layer.has_weights():
                            dW *= layer.W

                """ update """
                self.layers = pool.map(UpdateLayer(self.optimizer), gradients)

                print("loss on train set after iteration {}, batch {}: {}".format(epoch, batch_no, loss))
                batch_no += 1

            print("loss on validation set after iteration {}: {}".format(epoch, self.validate(X_valid, y_valid)))


    @staticmethod
    def store(self, model: 'Model', file_name: str) -> None:
        # TODO
        pass

    @staticmethod
    def load(self, file_name: str) -> 'Model':
        # TODO
        pass
