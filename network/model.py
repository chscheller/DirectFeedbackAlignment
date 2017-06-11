from typing import Iterable

import numpy as np
import time

from dataset.dataset import DataSet
from network.utils import data
from network.layer import Layer
from network.loss import SoftmaxCrossEntropyLoss, Loss
from network.optimizer import GDOptimizer, Optimizer


class UpdateLayer(object):
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def __call__(self, gradients: tuple) -> Layer:
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
            lr_decay: float=0,
            lr_decay_interval: int=0,
            regularization: float=0,
    ) -> None:
        self.layers = layers
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval
        self.regularization = regularization
        self.loss = loss
        self.statistics = {}
        self.__init_statistics()

    def __init_statistics(self):
        self.statistics = {
            'forward_time': 0,
            'regularization_time': 0,
            'backward_time': 0,
            'update_time': 0,
            'total_time': 0,
            'train_loss': [],
            'train_accuracy': [],
            'valid_step': [],
            'valid_loss': [],
            'valid_accuracy': [],
        }

    def cost(self, X, y):
        n = X.shape[0]

        """ forward pass """
        for layer in self.layers:
            X = layer.forward(X)

        """ loss """
        loss, _ = self.loss.calculate(X, y)
        accuracy = (np.argmax(X, axis=1) == y).sum() / y.shape[0]

        """ regularization (L2) """
        if self.regularization > 0:
            total_weights = 0
            for layer in self.layers:
                if layer.has_weights():
                    total_weights += np.sum(np.square(layer.W))
            loss += (total_weights * self.regularization / 2.) / n

        return loss, accuracy

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X, mode='train')
        return np.argmax(X, axis=1)

    def gradient_check(self, x, y):
        x_in = x

        """ forward pass """
        for layer in self.layers:
            x = layer.forward(x, mode='train')

        """ loss """
        loss, delta = self.loss.calculate(x, y)

        """ check gradients pass """
        # FIXME: hacky hack hack!
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
                    costPlus, _ = self.cost(x_in, y)
                    wMinus = W_unrolled.copy()
                    wMinus[i] -= epsilon
                    layer.W = wMinus.reshape(W_orig.shape)
                    costMinus, _ = self.cost(x_in, y)
                    dW_approx[i] = (costPlus - costMinus) / (2. * epsilon)
                layer.W = W_orig
                diff = np.linalg.norm(dW_approx - dW_unrolled)/np.linalg.norm(dW_approx + dW_unrolled)
                print("layer '{}', relative difference: {}".format(type(layer), diff))

    def train(self, data_set: DataSet, method: str, num_passes: int=20, batch_size: int=128, verbose: bool=True):

        if verbose:
            print(
                '\ntrain method: {}'.format(method),
                '\nnum_passes: {}'.format(num_passes),
                '\nbatch_size: {}\n'.format(batch_size)
            )

        start_total_time = time.time()

        X_train, y_train = data_set.train_set()
        X_valid, y_valid = data_set.validation_set()

        """ initalize layers """
        input_size = X_train[0].shape
        for layer in self.layers:
            input_size = layer.initialize(input_size, self.num_classes, method)
            layer.reset_params()

        step = 0
        for epoch in range(num_passes):

            """ decay learning rate if necessary """
            if self.lr_decay > 0 and epoch > 0 and (epoch % self.lr_decay_interval) == 0:
                self.optimizer.decay_learning_rate(self.lr_decay)
                if verbose:
                    print("Decreased learning rate by {}".format(self.lr_decay))

            for batch in data.mini_batches(X_train, y_train, batch_size):
                X_batch, y_batch = batch

                """ forward pass """
                start_forward_time = time.time()
                for layer in self.layers:
                    X_batch = layer.forward(X_batch, mode='train')
                self.statistics['forward_time'] += time.time() - start_forward_time

                """ loss """
                loss, delta = self.loss.calculate(X_batch, y_batch)

                """ backward pass """
                gradients = []
                start_backward_time = time.time()
                if method == 'dfa':
                    for layer in self.layers:
                        dW, db = layer.dfa(delta)
                        gradients.append((layer, dW, db))
                elif method == 'bp':
                    dX = delta
                    for layer in reversed(self.layers):
                        dX, dW, db = layer.back_prob(dX)
                        gradients.append((layer, dW, db))
                    gradients.reverse()
                else:
                    raise ValueError("Invalid train method '{}'".format(method))
                self.statistics['backward_time'] += time.time() - start_backward_time

                """ regularization (L2) """
                start_regularization_time = time.time()
                if self.regularization > 0:
                    reg_term = 0
                    for layer, dW, db in gradients:
                        if layer.has_weights():
                            dW += self.regularization * layer.W
                            reg_term += np.sum(np.square(layer.W))
                    reg_term *= self.regularization / 2.
                    reg_term /= y_batch.shape[0]
                    loss += reg_term
                self.statistics['regularization_time'] += time.time() - start_regularization_time

                """ update """
                start_update_time = time.time()
                update = UpdateLayer(self.optimizer)
                self.layers = [update(x) for x in gradients]
                self.statistics['update_time'] += time.time() - start_update_time

                """ log statistics """
                accuracy = (np.argmax(X_batch, axis=1) == y_batch).sum() / y_batch.shape[0]
                self.statistics['train_loss'].append(loss)
                self.statistics['train_accuracy'].append(accuracy)

                if (step % 10) == 0 and verbose:
                    print("epoch {}, step {}, loss = {:07.5f}, accuracy = {}".format(epoch, step, loss, accuracy))

                step += 1

            """ log statistics """
            valid_loss, valid_accuracy = self.cost(X_valid, y_valid)
            self.statistics['valid_step'].append(step)
            self.statistics['valid_loss'].append(valid_loss)
            self.statistics['valid_accuracy'].append(valid_accuracy)

            if verbose:
                print("validation after epoch {}: loss = {:07.5f}, accuracy = {}".format(epoch, valid_loss, valid_accuracy))

        self.statistics['total_time'] = time.time() - start_total_time
        return self.statistics

    @staticmethod
    def store(self, model: 'Model', file_name: str) -> None:
        # TODO
        pass

    @staticmethod
    def load(self, file_name: str) -> 'Model':
        # TODO
        pass