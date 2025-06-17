import matplotlib.pyplot as plt
import numpy as np


class BaseMetric:
    def __init__(self):
        super().__init__()

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def decent_loss(X: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: np.ndarray):
        raise NotImplementedError


class MSE(BaseMetric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.pow(y_true - y_pred, 2))

    @staticmethod
    def decent_loss(
        X: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray]:
        d_w = np.mean((-2 * X.T * (y_true - (X @ w + b))).T, axis=0)
        d_b = np.mean((X @ w + b) - y_true, axis=0)
        return d_w, d_b


class MAE(BaseMetric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def decent_loss(
        X: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray]:
        d_w = -1 * np.sign(y_true - (X @ w.T + b)) @ X
        d_b = np.mean(-1 * np.sign(y_true - (X @ w.T + b)), axis=0)
        return d_w, d_b


class Hists(BaseMetric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        plt.figure(figsize=(15, 10))
        plt.hist(y_true, label="True")
        plt.hist(y_pred, label="Prediction")
        plt.legend()
        plt.grid()
        plt.show()
