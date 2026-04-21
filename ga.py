"""Genetic algorithm with tournament selection, uniform crossover, and Gaussian mutation."""

import numpy as np


class GeneticAlgorithm:

    def __init__(self, pop_size: int, genome_size: int, tournament_size: int = 5,
                 crossover_rate: float = 0.7, mutation_rate: float = 0.1,
                 mutation_sigma: float = 0.1, elitism_count: int = 5):
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elitism_count = elitism_count

    def initialize(self, rng: np.random.Generator) -> np.ndarray:
        """Random initial population with small weights (≈ Xavier-like scale)."""
        return rng.standard_normal((self.pop_size, self.genome_size)) * 0.5

    def _tournament(self, population: np.ndarray, fitnesses: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        idxs = rng.integers(0, len(population), size=self.tournament_size)
        return population[idxs[np.argmax(fitnesses[idxs])]]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.crossover_rate:
            mask = rng.random(self.genome_size) < 0.5
            return np.where(mask, p1, p2)
        return p1.copy()

    def _mutate(self, genome: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(self.genome_size) < self.mutation_rate
        return genome + mask * rng.standard_normal(self.genome_size) * self.mutation_sigma

    def next_generation(self, population: np.ndarray, fitnesses: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        """Produce the next generation: elitism → tournament → crossover → mutation."""
        ranked = np.argsort(fitnesses)[::-1]
        new_pop = [population[ranked[i]].copy() for i in range(self.elitism_count)]

        while len(new_pop) < self.pop_size:
            p1 = self._tournament(population, fitnesses, rng)
            p2 = self._tournament(population, fitnesses, rng)
            child = self._crossover(p1, p2, rng)
            child = self._mutate(child, rng)
            new_pop.append(child)

        return np.array(new_pop[:self.pop_size])
