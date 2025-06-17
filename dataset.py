import matplotlib.pyplot as plt


class Dataset:
    def __init__(self):
        self.X = None
        self.y = None

    def show(self):
        plt.figure(figsize=(15, 4))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, alpha=0.8)
