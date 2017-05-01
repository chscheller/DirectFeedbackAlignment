import numpy as np


def slice_mini_batches(X, y, batch_size):
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        indices = slice(start_idx, start_idx + batch_size)
        yield X[indices], y[indices]


def split_train_valid_data(X, y, train_ratio=0.8):
    train_size = int(X.shape[0] * train_ratio)
    y_train = np.take(y, range(int(train_size)), axis=0)
    X_train = np.take(X, range(int(train_size)), axis=0)
    y_valid = np.take(y, range(train_size, X.shape[0]), axis=0)
    X_valid = np.take(X, range(train_size, X.shape[0]), axis=0)
    return y_train, X_train, y_valid, X_valid