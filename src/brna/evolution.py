"""Clean-room genetic optimization for agent policy weights."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class EvolutionTrainer:
    def __init__(
        self,
        *,
        population_size: int = 32,
        elite_fraction: float = 0.2,
        mutation_scale: float = 0.1,
        seed: int = 0,
    ):
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not math.isfinite(elite_fraction) or not 0 < elite_fraction <= 1:
            raise ValueError("elite_fraction must be between 0 and 1")
        if not math.isfinite(mutation_scale) or mutation_scale < 0:
            raise ValueError("mutation_scale must be finite and non-negative")
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_scale = mutation_scale
        self._rng = np.random.default_rng(seed)

    def initial_population(self, genome_shape: int | tuple[int, ...]) -> list[np.ndarray]:
        if isinstance(genome_shape, int):
            if genome_shape < 1:
                raise ValueError("genome size must be positive")
            shape = (genome_shape,)
        else:
            shape = tuple(genome_shape)
            if not shape or any(size < 1 for size in shape):
                raise ValueError("genome dimensions must be positive")
        return [
            self._rng.normal(0.0, 1.0, size=shape)
            for _ in range(self.population_size)
        ]

    def next_generation(
        self,
        population: Sequence[np.ndarray],
        fitnesses: Sequence[float],
    ) -> list[np.ndarray]:
        if len(population) != self.population_size or len(fitnesses) != len(population):
            raise ValueError("population and fitnesses must match population_size")
        genomes = [np.asarray(genome, dtype=float) for genome in population]
        if not genomes or any(genome.shape != genomes[0].shape for genome in genomes):
            raise ValueError("all genomes must have the same non-empty shape")
        if any(genome.size == 0 or not np.isfinite(genome).all() for genome in genomes):
            raise ValueError("genomes must contain finite values")
        scores = np.asarray(fitnesses, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("fitnesses must be finite")

        elite_count = max(1, math.ceil(self.population_size * self.elite_fraction))
        ranked = np.argsort(-scores, kind="stable")
        elites = [genomes[index].copy() for index in ranked[:elite_count]]
        next_population = [genome.copy() for genome in elites]
        while len(next_population) < self.population_size:
            parent_a, parent_b = self._rng.choice(elite_count, size=2, replace=True)
            mask = self._rng.random(size=genomes[0].shape) < 0.5
            child = np.where(mask, elites[parent_a], elites[parent_b]).copy()
            child += self._rng.normal(0.0, self.mutation_scale, size=child.shape)
            next_population.append(child)
        return next_population
