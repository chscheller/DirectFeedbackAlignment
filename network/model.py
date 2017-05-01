import multiprocessing
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import time

from network.utils import data
from network.layer import Layer
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
            input_size: tuple,
            num_classes: int,
            loss: Loss=SoftmaxCrossEntropyLoss(),
            optimizer: Optimizer=GDOptimizer(),
            lr_decay: float=0,
            lr_decay_interval: int=0,
            regularization: float=0,
            method='dfa'
    ) -> None:
        self.layers = layers
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval
        self.regularization = regularization
        self.loss = loss
        self.method = method

        """ initalize layers """
        for layer in self.layers:
            input_size = layer.initialize(input_size, self.num_classes, method)

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
        X -= self.mean
        X /= self.std
        for layer in self.layers:
            X = layer.forward(X, mode='train')
        return np.argmax(X, axis=1)

    def gradient_check(self, x, y):
        x -= self.mean
        x /= self.std
        x_in = x

        """ forward pass """
        for layer in self.layers:
            x = layer.forward(x, mode='train')

        """ loss """
        loss, delta = self.loss.calculate(x, y)

        """ check gradients pass """
        if self.method == 'bp':
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
                    print("layer '{}', relative difference: {}".format(
                        type(layer),
                        np.linalg.norm(dW_approx - dW_unrolled)/np.linalg.norm(dW_approx + dW_unrolled))
                    )
        else:
            raise (Exception("no gradient check available for train method '{}'".format(self.method)))


    def train(self, X, y, num_passes=2000, batch_size=128):
        y_train, X_train, y_valid, X_valid = data.split_train_valid_data(X, y)

        self.mean = np.mean(X_train, axis=0)
        X_train -= self.mean
        X_valid -= self.mean

        self.std = np.std(X_train, axis=0)
        X_train /= self.std
        X_valid /= self.std

        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        forward_time = 0
        backward_time = 0
        total_time = 0

        start_total_time = time.time()

        train_losses = []
        cv_losses = []

        for epoch in range(num_passes):
            batch_no = 0
            for batch in data.slice_mini_batches(X_train, y_train, batch_size):
                X_batch, y_batch = batch

                """ forward pass """
                start_forward_time = time.time()
                for layer in self.layers:
                    X_batch = layer.forward(X_batch, mode='train')
                forward_time += time.time() - start_forward_time

                """ loss """
                loss, delta = self.loss.calculate(X_batch, y_batch)

                """ backward pass """
                gradients = []
                start_backward_time = time.time()
                if self.method == 'dfa':
                    # gradients = pool.map(DFA(delta), self.layers)
                    for layer in reversed(self.layers):
                        dW, db = layer.dfa(delta)
                        gradients.append((layer, dW, db))
                else:
                    dX = delta
                    for layer in reversed(self.layers):
                        dX, dW, db = layer.back_prob(dX)
                        gradients.append((layer, dW, db))
                    gradients.reverse()
                backward_time += time.time() - start_backward_time

                """ regularization (L2) """
                if self.regularization > 0:
                    for layer, dW, db in gradients:
                        if layer.has_weights():
                            dW += self.regularization * layer.W

                """ update """
                # self.layers = pool.map(UpdateLayer(self.optimizer), gradients)
                # self.layers = map(UpdateLayer(self.optimizer), gradients)
                optim = UpdateLayer(self.optimizer)
                for gradient in gradients:
                    optim(gradient)

                print("loss on train set after iteration {}, batch {}: {}".format(epoch, batch_no, loss))
                train_losses.append(loss)
                #print("{}".format(loss))
                batch_no += 1

            cost = self.cost(X_valid, y_valid)
            print("loss on validation set after iteration {}: {}".format(epoch, cost))
            cv_losses.append(cost)
            #print("{}".format(self.cost(X_valid, y_valid)))

            if self.lr_decay > 0 and epoch > 0 and (epoch % self.lr_decay_interval) == 0:
                self.optimizer.decay_learning_rate(self.lr_decay)

        total_time = time.time() - start_total_time
        print("time spend during forward pass: {}".format(forward_time))
        print("time spend during backward pass: {}".format(backward_time))
        print("time spend in total: {}".format(total_time))

        plt.plot(range(len(train_losses)), train_losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Train loss')
        plt.grid(True)
        plt.show()

        plt.plot(range(len(cv_losses)), cv_losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('CV loss')
        plt.grid(True)
        plt.show()

    def test(self, X, y):
        X -= self.mean
        X /= self.std
        prediction = self.predict(X)
        n = X.shape[0]
        correct = (prediction == y).sum()
        print("accuracy: {}".format(correct/n))

    @staticmethod
    def store(self, model: 'Model', file_name: str) -> None:
        # TODO
        pass

    @staticmethod
    def load(self, file_name: str) -> 'Model':
        # TODO
        pass
