import os
import struct
import numpy as np
from array import array

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


def load(path: str='mnist', valid_size: int=5000):
    images_train_valid = load_images(os.path.join(path, train_images_filename))
    labels_train_valid = load_labels(os.path.join(path, train_labels_filename))

    images_valid = images_train_valid[:valid_size]
    labels_valid = labels_train_valid[:valid_size]

    images_train = images_train_valid[valid_size:]
    labels_train = labels_train_valid[valid_size:]

    images_test = load_images(os.path.join(path, test_images_filename))
    labels_test = load_labels(os.path.join(path, test_labels_filename))

    # mean subtraction
    train_mean = np.mean(images_train, axis=0)
    images_train -= train_mean
    images_valid -= train_mean
    images_test -= train_mean

    # normalization
    train_std = np.std(images_train, axis=0)
    train_std += (train_std == 0).astype(int)
    images_train /= train_std
    images_valid /= train_std
    images_test /= train_std





class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.train_images = []
        self.train_labels = []

        self.valid_images = []
        self.valid_labels = []

        self.test_images = []
        self.test_labels = []

        self.load_train_valid()
        self.load_test()

    def load_train_valid(self, valid_size=5000):
        images, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        images, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                   os.path.join(self.path, self.train_lbl_fname))
        images = np.asarray(images)
        images = images.reshape((images.shape[0], 1, 28, 28))
        images = np.asarray(images)

        images_test, labels_test = mnist.load_training()
        X_test = np.asarray(images_test)
        X_test = X.reshape((X_test.shape[0], 1, 28, 28))
        y_test = labels_test

        self.train_images = images
        self.train_labels = labels

        return ims, labels

    def load_test(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))



        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels