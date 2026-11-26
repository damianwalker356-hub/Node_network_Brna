"""Rev B-style toy message-passing network and loop-control primitives."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Critic:
    hop_limit: int = 3
    budget: int = 32
    seen: set[tuple[int, int]] = field(default_factory=set)
    count: int = 0
    def allow(self, source: int, target: int, hop: int) -> bool:
        edge = (source, target)
        if hop >= self.hop_limit or self.count >= self.budget or source == target or edge in self.seen:
            return False
        self.seen.add(edge); self.count += 1
        return True

class ToyNetwork:
    def __init__(self, adjacency: np.ndarray, gates: np.ndarray | None = None, message_bound: float = 1.0, budget: int = 32):
        self.adjacency = np.asarray(adjacency, dtype=float)
        if self.adjacency.ndim != 2 or self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError('adjacency must be square')
        self.gates = np.zeros_like(self.adjacency) if gates is None else np.asarray(gates, dtype=float)
        if self.gates.shape != self.adjacency.shape: raise ValueError('gate shape mismatch')
        self.message_bound, self.budget = float(message_bound), int(budget)
    def bounded_message(self, message: np.ndarray) -> np.ndarray:
        x = np.asarray(message, dtype=float); norm = np.linalg.norm(x)
        return x if norm <= self.message_bound or norm == 0 else x * (self.message_bound / norm)
    def gate_matrix(self) -> np.ndarray:
        return self.adjacency * (1.0 / (1.0 + np.exp(-np.clip(self.gates, -40, 40))))
    def exchange(self, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=float); result = self.gate_matrix() @ x
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        return result * np.minimum(1.0, self.message_bound / np.maximum(norms, 1e-12))
    def propagate(self, states: np.ndarray, hops: int = 3) -> np.ndarray:
        if hops < 0: raise ValueError('hops must be non-negative')
        x = np.asarray(states, dtype=float).copy()
        for _ in range(min(hops, 3)): x = np.tanh(x + self.exchange(x))
        return x
