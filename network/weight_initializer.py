import numpy as np


class WeightInitializer(object):
    def init(self, dim: tuple) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class Fill(WeightInitializer):

    def __init__(self, fill_value: float) -> None:
        self.fill_value = fill_value

    def init(self, dim: tuple) -> np.ndarray:
        return np.full(shape=dim, fill_value=self.fill_value, dtype=float)

    def __str__(self):
        return "Fill(fill_value={})".format(self.fill_value)


class RandomUniform(WeightInitializer):

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def init(self, dim: tuple) -> np.ndarray:
        return np.random.uniform(low=self.low, high=self.high, size=dim)

    def __str__(self):
        return "Uniform(low={}, high={})".format(self.low, self.high)


class RandomNormal(WeightInitializer):

    def __init__(self, sigma: float=1, mu: float= 0) -> None:
        self.sigma = sigma
        self.mu = mu

    def init(self, dim: tuple) -> np.ndarray:
        return self.sigma * np.random.randn(*dim) + self.mu

    def __str__(self):
        return "Normal(sigma={}, mu={})".format(self.sigma, self.mu)