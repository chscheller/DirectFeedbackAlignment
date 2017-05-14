import numpy as np


def mini_batches(X, y, batch_size, shuffle=True):
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, X.shape[0] - batch_size + 1, batch_size):
        curr_indices = indices[i:i + batch_size]
        yield X[curr_indices], y[curr_indices]