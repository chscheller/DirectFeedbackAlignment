import numpy as np


class Layer:

    def __init__(self) -> None:
        super().__init__()
        self.params = dict()

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

    def get_params(self) -> dict:
        return self.params

    def get_param(self, key: any) -> any:
        return self.params[key]

    def set_param(self, key: any, value) -> None:
        self.params[key] = value

    def has_param(self, key: any) -> bool:
        return key in self.params.keys()
