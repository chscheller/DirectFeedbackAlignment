import os
import struct
import numpy as np
from array import array

from dataset.dataset import DataSet

train_images_filename = 'train-images-idx3-ubyte'
train_labels_filename = 'train-labels-idx1-ubyte'

test_images_filename = 't10k-images-idx3-ubyte'
test_labels_filename = 't10k-labels-idx1-ubyte'


def load_images(full_path):
    with open(full_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Invalid magic number in MNIST image file: expected {} got {}'.format(2051, magic))
        raw_data = array("B", file.read())
        images = np.zeros((size, 1, rows * cols))
        image_size = rows * cols
        for i in range(size):
            images[i, 0, :] = raw_data[i * image_size:(i + 1) * image_size]
        return images.reshape(size, 1, rows, cols)


def load_labels(full_path):
    with open(full_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Invalid magic number in MNIST label file: expected {} got {}'.format(2049, magic))
        labels = array("B", file.read())
        return np.asarray(labels)


def load(path: str='mnist', valid_size: int=5000, mean_subtraction=True, normalization=True):
    images_train_valid = load_images(os.path.join(path, train_images_filename))
    labels_train_valid = load_labels(os.path.join(path, train_labels_filename))

    images_valid = images_train_valid[:valid_size]
    labels_valid = labels_train_valid[:valid_size]

    images_train = images_train_valid[valid_size:]
    labels_train = labels_train_valid[valid_size:]

    images_test = load_images(os.path.join(path, test_images_filename))
    labels_test = load_labels(os.path.join(path, test_labels_filename))

    # mean subtraction
    if mean_subtraction:
        train_mean = np.mean(images_train, axis=0)
        images_train -= train_mean
        images_valid -= train_mean
        images_test -= train_mean

    # normalization
    if normalization:
        train_std = np.std(images_train, axis=0)
        train_std += (train_std == 0).astype(int)
        images_train /= train_std
        images_valid /= train_std
        images_test /= train_std

    return DataSet(
        train=(images_train, labels_train),
        validation=(images_valid, labels_valid),
        test=(images_test, labels_test)
    )
