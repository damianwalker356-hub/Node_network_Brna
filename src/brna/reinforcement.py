"""Small tabular Q-learning primitive for discrete agent environments."""

from __future__ import annotations

import math

import numpy as np


class QLearner:
    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        seed: int = 0,
    ):
        for name, value in (
            ("learning_rate", learning_rate),
            ("discount", discount),
            ("epsilon", epsilon),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_values: dict[str, dict[str, float]] = {}
        self._rng = np.random.default_rng(seed)

    def choose_action(self, state: str, actions: list[str]) -> str:
        if not actions:
            raise ValueError("at least one action is required")
        if self._rng.random() < self.epsilon:
            return actions[int(self._rng.integers(len(actions)))]
        values = self.q_values.get(state, {})
        return max(actions, key=lambda action: values.get(action, 0.0))

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_actions: list[str],
        *,
        terminal: bool = False,
    ) -> float:
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        current = self.q_values.setdefault(state, {}).get(action, 0.0)
        future = (
            max((self.q_values.get(next_state, {}).get(item, 0.0) for item in next_actions), default=0.0)
            if not terminal
            else 0.0
        )
        target = reward + self.discount * future
        updated = current + self.learning_rate * (target - current)
        self.q_values[state][action] = updated
        return updated

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "q_values": {state: dict(values) for state, values in self.q_values.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, *, seed: int = 0) -> QLearner:
        learner = cls(
            learning_rate=data["learning_rate"],
            discount=data["discount"],
            epsilon=data["epsilon"],
            seed=seed,
        )
        q_values = data["q_values"]
        if not isinstance(q_values, dict):
            raise TypeError("q_values must be a mapping")
        for state, values in q_values.items():
            if not isinstance(state, str) or not isinstance(values, dict):
                raise TypeError("Q-table states and action values have invalid shape")
            for action, value in values.items():
                if (
                    not isinstance(action, str)
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                ):
                    raise TypeError("Q-table values must be numbers")
                if not math.isfinite(value):
                    raise ValueError("Q-table values must be finite")
        learner.q_values = {state: dict(values) for state, values in q_values.items()}
        return learner
