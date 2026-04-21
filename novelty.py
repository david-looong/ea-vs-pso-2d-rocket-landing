"""
Novelty archive for behavioural diversity.

Each genome gets a 5-D behaviour descriptor after evaluation:
  (avg altitude, max tilt, fuel used, time to ground, horiz distance).

The novelty score is the mean distance to the k nearest neighbours in the
combined archive + current population.  A bonus proportional to this score
is added to task fitness so the GA rewards *unique* strategies, not just raw
performance.  This keeps the population from collapsing to 200 copies of the
same policy.
"""

import numpy as np


class NoveltyArchive:

    def __init__(self, k: int = 15, archive_prob: float = 0.05):
        self.k = k
        self.archive_prob = archive_prob
        self.archive: list[np.ndarray] = []

    def compute_novelty(self, behaviours: np.ndarray) -> np.ndarray:
        """Return per-genome novelty scores (higher = more novel)."""
        n = len(behaviours)
        if n < 2:
            return np.zeros(n)

        pool = (np.vstack([behaviours, np.array(self.archive)])
                if self.archive else behaviours.copy())

        # unit-variance normalisation so all descriptor dimensions matter equally
        mean = pool.mean(axis=0)
        std = pool.std(axis=0) + 1e-8
        pool_n = (pool - mean) / std
        pop_n = pool_n[:n]

        scores = np.zeros(n)
        for i in range(n):
            dists = np.sqrt(np.sum((pool_n - pop_n[i]) ** 2, axis=1))
            dists.sort()
            knn = dists[1:self.k + 1]  # skip self (dist = 0)
            scores[i] = knn.mean() if len(knn) > 0 else 0.0
        return scores

    def update(self, behaviours: np.ndarray, rng: np.random.Generator) -> None:
        """Probabilistically archive behaviours for long-term diversity memory."""
        for b in behaviours:
            if rng.random() < self.archive_prob:
                self.archive.append(b.copy())
