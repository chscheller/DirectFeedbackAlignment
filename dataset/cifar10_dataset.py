import os
import numpy as np

from dataset.dataset import DataSet

train_files = 5
test_files = 1

train_size = 50000
test_size = 10000
channels = 3
image_height = 32
image_width = 32


def unpickle(file_path: str) -> dict:
    import pickle
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data


def load_data(full_path):
    raw_data = unpickle(full_path)
    float_data = np.array(raw_data[b'data'], dtype=float) / 255.0
    images = float_data.reshape([-1, channels, image_height, image_width])
    labels = np.array(raw_data[b'labels'])
    return images, labels


def load_test_set(self) -> (np.ndarray, np.ndarray):
    file_path = os.path.join(self.data_path, "test_batch")
    images, labels = self.__load_file(file_path)
    return images, labels


def load(path: str='dataset/cifar10/cifar-10-batches-py', valid_size: int=5000, mean_subtraction=True, normalization=True):

    images_train_valid = np.zeros(shape=(train_size, channels, image_height, image_width), dtype=float)
    labels_train_valid = np.zeros(shape=train_size, dtype=int)

    start = 0
    for i in range(train_files):
        images_batch, labels_batch = load_data(os.path.join(path, "data_batch_{}".format(i + 1)))
        size = images_batch.shape[0]
        images_train_valid[start:start + size] = images_batch
        labels_train_valid[start:start + size] = labels_batch
        start = start + size

    images_valid = images_train_valid[:valid_size]
    labels_valid = labels_train_valid[:valid_size]

    images_train = images_train_valid[valid_size:]
    labels_train = labels_train_valid[valid_size:]

    images_test, labels_test = load_data(os.path.join(path, "test_batch"))

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
