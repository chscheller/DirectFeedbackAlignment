import multiprocessing
import numpy as np
import time


class CallDFA(object):
    def __init__(self, e, reg, lr):
        self.e = e
        self.reg = reg
        self.lr = lr

    def __call__(self, layer):
        return layer.dfa(self.e, self.reg, self.lr)


class Network(object):
    def __init__(self, layers):
        """
        :param layers:
        :type layers: Layer[]
        """
        self.layers = layers
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())

    @staticmethod
    def slice_mini_batches(X, y, batch_size):
        for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
            indices = slice(start_idx, start_idx + batch_size)
            yield X[indices], y[indices]

    @staticmethod
    def split_train_data(X, y, train_ratio=0.8):
        train_size = int(X.shape[0] * train_ratio)
        y_train = np.take(y, range(int(train_size)), axis=0)
        X_train = np.take(X, range(int(train_size)), axis=0)
        y_valid = np.take(y, range(train_size, X.shape[0]), axis=0)
        X_valid = np.take(X, range(train_size, X.shape[0]), axis=0)
        return y_train, X_train, y_valid, X_valid

    def loss(self, X, y, reg):
        train_size = X.shape[0]
        for layer in self.layers:
            X = layer.forward(X)
        correct_logprobs = -np.log(X[range(train_size), y])
        data_loss = np.sum(correct_logprobs)
        total_weight = 0
        for layer in self.layers:
            total_weight += layer.sum_weights()
        data_loss += reg / 2 * total_weight
        return 1. / train_size * data_loss

    def train(self, X, y, reg, lr, num_passes=2000, batch_size=128, method='dfa', print_loss=True):
        y_train, X_train, y_valid, X_valid = self.split_train_data(X, y)

        out_layer_size = 10  # TODO: determine by looking at last layer!
        input_size = (X.shape[1], X.shape[2], X.shape[3])
        for layer in self.layers:
            input_size = layer.initialize(input_size, out_layer_size, method)

        total_time = 0
        forward_time = 0
        update_time = 0
        for epoch in range(num_passes):
            start_total = time.time()
            for batch in self.slice_mini_batches(X_train, y_train, batch_size):
                X_batch, y_batch = batch
                train_size = X_batch.shape[0]

                start = time.time()
                for layer in self.layers:
                    X_batch = layer.forward(X_batch)
                probs = X_batch
                forward_time += time.time() - start

                start = time.time()
                e = probs
                e[range(train_size), y_batch] -= 1
                if method == 'dfa':
                    self.layers = self.pool.map(CallDFA(e, reg, lr), self.layers)
                    # print(res)
                    # self.pool.join()
                    # e = X_batch
                    # for layer in self.layers:
                    #     layer.dfa(e, reg, lr)
                else:
                    delta = X_batch
                    for layer in reversed(self.layers):
                        delta = layer.back_prob(delta, reg, lr)
                update_time += time.time() - start

            total_time += time.time() - start_total

            if print_loss and epoch % 1 == 0:
                print("Loss after iteration {}: {}".format(epoch, self.loss(X_valid, y_valid, reg)))

        print("Time spent in forward loop: {}s".format(forward_time))
        print("Time spent in update loop: {}s".format(update_time))
        print("Time spent in total: {}s".format(total_time))

    def test(self, X, y):
        prediction = self.predict(X)
        total = X.shape[0]
        correct = (prediction == y).sum()
        print("accuracy: {}".format(correct/total))

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x, axis=1)
