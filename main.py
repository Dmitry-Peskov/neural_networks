import numpy as np


class Function:

    @staticmethod
    def sigmoid(x: int | float):
        return 1 / (1 + np.exp(-x))


class Neuron:

    def __init__(
            self,
            weights: list | np.ndarray,
            bias: int | float
    ):
        self.weights = weights
        self.bias = bias

    def feedforward(self, input) -> float:
        total = np.dot(self.weights, input) + self.bias
        return Function.sigmoid(total)


if __name__ == "__main__":
    n = Neuron(np.array([0, 1]), 4)
    x = np.array([4,5])
    print(n.feedforward(x))