import os

import numpy as np


class Cifar10(object):
    def __init__(self, path: str='.', data_set: str='cifar-10-batches-py') -> None:
        self.path = path
        self.data_set = data_set
        self.data_path = os.path.join(path, data_set)

        self.train_files = 5
        self.test_files = 1

        self.train_size = 50000
        self.test_size = 10000
        self.channels = 3
        self.image_height = 32
        self.image_width = 32

    @staticmethod
    def __unpickle(file_path: str) -> dict:
        import pickle
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
        return data

    @staticmethod
    def __one_hot_encode(data: np.ndarray) -> np.ndarray:
        n = np.max(data) + 1
        return np.eye(n)[data]

    def __load_file(self, file_path: str) -> (np.ndarray, np.ndarray):
        raw_data = self.__unpickle(file_path)
        float_data = np.array(raw_data[b'data'], dtype=float) / 255.0
        images = float_data.reshape([-1, self.channels, self.image_height, self.image_width])
        labels = np.array(raw_data[b'labels'])
        return images, labels

    def load_train_set(self) -> (np.ndarray, np.ndarray):
        images = np.zeros(shape=(self.train_size, self.channels, self.image_height, self.image_width), dtype=float)
        labels = np.zeros(shape=self.train_size, dtype=int)
        start = 0
        for i in range(self.train_files):
            file_path = os.path.join(self.data_path, "data_batch_".format(i + 1))
            images_batch, labels_batch = self.__load_file(file_path)
            size = images.shape[0]
            images[start:start + size] = images_batch
            labels[start:start + size] = labels_batch
            start = start + size
        return images, labels

    def load_test_set(self) -> (np.ndarray, np.ndarray):
        file_path = os.path.join(self.data_path, "test_batch")
        images, labels = self.__load_file(file_path)
        return images, labels

    def load_labels(self) -> [str]:
        file_path = os.path.join(self.data_path, "batches.meta")
        raw_data = self.__unpickle(file_path)
        return map(lambda raw_data: raw_data.decode('utf-8'), raw_data[b'label_names'])
