"""Feed-forward neural-network rocket controller (NumPy only)."""

import numpy as np


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


class NeuralNetwork:
    """
    Small fully-connected network: 6 → 16 → 16 → 2.

    * Hidden activations: tanh
    * Output[0] → throttle  (sigmoid → [0, 1])
    * Output[1] → gimbal    (tanh   → [-1, 1])

    Genome = flat float vector of all weights and biases.
    """

    def __init__(self, layers: list[int], genome: np.ndarray | None = None):
        self.layers = layers
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._build(genome)

    @staticmethod
    def genome_size(layers: list[int]) -> int:
        return sum(layers[i] * layers[i + 1] + layers[i + 1]
                   for i in range(len(layers) - 1))

    def _build(self, genome: np.ndarray | None) -> None:
        self.weights, self.biases = [], []
        if genome is None:
            for i in range(len(self.layers) - 1):
                self.weights.append(np.zeros((self.layers[i], self.layers[i + 1])))
                self.biases.append(np.zeros(self.layers[i + 1]))
            return
        offset = 0
        for i in range(len(self.layers) - 1):
            n_in, n_out = self.layers[i], self.layers[i + 1]
            w_size = n_in * n_out
            self.weights.append(genome[offset:offset + w_size].reshape(n_in, n_out))
            offset += w_size
            self.biases.append(genome[offset:offset + n_out].copy())
            offset += n_out

    def set_genome(self, genome: np.ndarray) -> None:
        self._build(genome)

    def forward(self, x: np.ndarray) -> tuple[float, float]:
        """Return (throttle ∈ [0,1], gimbal ∈ [-1,1])."""
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ w + b
            if i < len(self.weights) - 1:
                a = np.tanh(a)
        throttle = _sigmoid(float(a[0]))
        gimbal = float(np.tanh(a[1]))
        return throttle, gimbal
