import numpy as np

from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from activations import Sigmoid
from optimizers import BaseOptimizer
from utils import GifMaker


class BaseClassifier:
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class LogReg(BaseClassifier):
    def __init__(self, optimizer: BaseOptimizer, lr: float = 1.0):
        super().__init__()
        self.gm = GifMaker(duration=0.4)
        self.optimizer = optimizer
        self.output_cls = Sigmoid()
        self.lr = lr
        self.weights = None
        self.bias = None

    def update_weights(self, d_w: np.ndarray, d_b: np.ndarray) -> None:
        self.weights -= d_w * self.lr
        self.bias -= d_b * self.lr
        return np.linalg.norm(d_w)

    def get_decent(
        self, X: np.ndarray, y_t: np.ndarray, w: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray]:
        return self.optimizer.get_gradient(X, y_t, w, b)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int = 1000,
        gradient_threshold: float = 0.0001,
        verbose: bool = True,
        test_size: float = 0.0,
    ):
        if test_size:
            loss_history_test = []
            X, X_test, y, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=42
            )

        self.gm.images = []
        loss_history = []
        for iter in tqdm(range(1, max_iter)):
            prediction = self.predict_proba(X)
            loss_history.append(self.optimizer.loss_cls.loss(y, prediction))
            weights_norm = self.update_weights(*self.get_decent(X, y, self.weights, self.bias))

            if test_size:
                loss_history_test.append(self.optimizer.loss_cls.loss(y_test, self.predict(X_test)))

            if verbose == "plot":
                self.gm.add_frames(loss_history, loss_history_test, self.weights)

            elif verbose == "print":
                logger.info(f"Train Loss: {loss_history[-1]}")

            if weights_norm < gradient_threshold or weights_norm > 10_000_000:
                break

        if verbose == "plot":
            self.gm.create_gif()

    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            self.weights = np.random.random(size=X.shape[1])
            self.bias = np.ones(shape=(1,))

        logger.debug(f"Weights: {self.weights}, Bias: {self.bias}")
        return self.output_cls(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        return self.predict_proba(X) > threshold
