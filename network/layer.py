import numpy as np


class Layer:

    def initialize(self, input_size: tuple, num_classes: int, train_method: str) -> tuple:
        pass

    def forward(self, X: np.ndarray, mode='predict') -> np.ndarray:
        pass

    def dfa(self, E: np.ndarray) -> tuple:
        pass

    def back_prob(self, E: np.ndarray) -> tuple:
        pass

    def has_weights(self) -> bool:
        return False
