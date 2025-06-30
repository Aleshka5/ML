import numpy as np


class BaseActivation:
    def __init__(self):
        pass

    def __call__(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decent(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(BaseActivation):
    def __init__(self):
        pass

    @staticmethod
    def _sigmoid(y: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-y))

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self._sigmoid(y)

    def decent(self, y: np.ndarray) -> np.ndarray:
        return self._sigmoid(y) - (1 - self._sigmoid(y))


class Relu(BaseActivation):
    def __init__(self):
        pass
    
    @staticmethod
    def _relu(y: np.ndarray) -> np.ndarray:
        return np.max(0, y)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self._relu(y)

    def decent(self, y: np.ndarray) -> np.ndarray:
        return 1 if y > 0 else 0
