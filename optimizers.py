import numpy as np

from metrics import MSE, BaseMetric


class BaseOptimizer:
    def __init__(self):
        self.loss_cls = BaseMetric()

    def get_gradient(self, X, y, w, b):
        raise NotImplementedError


class FullGD(BaseOptimizer):
    def __init__(self, loss=MSE(), a: float = 0.0, penalty: str = "l2"):
        self.loss_cls = loss
        self.penalty = penalty
        self.a = a

    def get_gradient(self, X, y, w, b) -> tuple[np.ndarray]:
        d_w, d_b = self.loss_cls.decent_loss(X, y, w, b)
        if self.penalty == "l2":
            d_w += self.a * 2 * w
        elif self.penalty == "l1":
            d_w += self.a * np.sign(w)
        return d_w, d_b


class SGD(BaseOptimizer):
    def __init__(
        self,
        loss=MSE(),
        batch_size: int = 64,
        a: float = 0.0,
        S0: float = 1.0,
        p: float = 0.2,
        penalty: str = "l2",
    ):
        self.loss_cls = loss
        self.batch_size = batch_size
        self.penalty = penalty
        self.a = a
        self.S0 = S0
        self.p = p
        self.cur_step = 0

    def get_gradient(self, X, y, w, b) -> tuple[np.ndarray]:
        self.cur_step += 1

        batch_indexes = np.random.choice(X.shape[0], self.batch_size, replace=False)
        d_w, d_b = self.loss_cls.decent_loss(X[batch_indexes], y[batch_indexes], w, b)

        if self.penalty == "l2":
            d_w += self.a * 2 * w

        elif self.penalty == "l1":
            d_w += self.a * np.sign(w)

        d_w *= np.pow((self.S0 / (self.S0 + self.cur_step)), self.p)
        return d_w, d_b
