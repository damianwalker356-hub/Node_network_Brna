"""Original, minimal creature-foraging environment for policy experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from brna.agents import AgentDefinition
from brna.reinforcement import QLearner

ACTIONS = ("left", "right", "eat", "rest")


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    steps: int
    food_eaten: int


class CreatureEnvironment:
    """A bounded one-dimensional foraging task with local observations."""

    def __init__(
        self,
        *,
        length: int = 9,
        food_count: int = 3,
        max_steps: int = 40,
        food_positions: Sequence[int] | None = None,
    ):
        if length < 3:
            raise ValueError("length must be at least 3")
        if not 1 <= food_count < length:
            raise ValueError("food_count must be between 1 and length - 1")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.length = length
        self.food_count = food_count
        self.max_steps = max_steps
        self.fixed_food = None if food_positions is None else frozenset(food_positions)
        if self.fixed_food is not None:
            start = length // 2
            if (
                len(self.fixed_food) != food_count
                or any(position < 0 or position >= length for position in self.fixed_food)
                or start in self.fixed_food
            ):
                raise ValueError("food positions must be unique valid cells away from the start")
        self.reset(seed=0)

    def reset(self, *, seed: int = 0) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        self.position = self.length // 2
        candidates = [cell for cell in range(self.length) if cell != self.position]
        self.food = set(
            self.fixed_food
            if self.fixed_food is not None
            else self._rng.choice(candidates, size=self.food_count, replace=False).tolist()
        )
        self.energy = 1.0
        self.steps = 0
        self.food_eaten = 0
        self.done = False
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.array(
            [
                self.energy,
                float(self.position in self.food),
                float(self.position > 0 and self.position - 1 in self.food),
                float(self.position < self.length - 1 and self.position + 1 in self.food),
            ],
            dtype=float,
        )

    def step(self, action: str) -> tuple[np.ndarray, float, bool]:
        if self.done:
            raise RuntimeError("episode has ended; reset the environment before stepping")
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action!r}")

        reward = -0.01
        self.energy = max(0.0, self.energy - 0.03)
        if action == "left":
            if self.position > 0:
                self.position -= 1
            else:
                reward -= 0.05
        elif action == "right":
            if self.position < self.length - 1:
                self.position += 1
            else:
                reward -= 0.05
        elif action == "eat":
            if self.position in self.food:
                self.food.remove(self.position)
                self.food_eaten += 1
                self.energy = min(1.0, self.energy + 0.45)
                reward += 1.0
            else:
                reward -= 0.1
        elif action == "rest":
            self.energy = min(1.0, self.energy + 0.05)

        self.steps += 1
        self.done = (
            self.energy <= 0.0
            or self.steps >= self.max_steps
            or not self.food
        )
        return self.observe(), reward, self.done


def run_policy_episode(
    agent: AgentDefinition,
    environment: CreatureEnvironment,
    *,
    seed: int = 0,
) -> EpisodeResult:
    observation = environment.reset(seed=seed)
    total_reward = 0.0
    while not environment.done:
        action = agent.choose_action(observation, ACTIONS)
        observation, reward, _ = environment.step(action)
        total_reward += reward
    return EpisodeResult(total_reward, environment.steps, environment.food_eaten)


def run_q_episode(
    learner: QLearner,
    environment: CreatureEnvironment,
    *,
    seed: int = 0,
) -> EpisodeResult:
    observation = environment.reset(seed=seed)
    total_reward = 0.0
    while not environment.done:
        state = _state_key(observation)
        action = learner.choose_action(state, list(ACTIONS))
        next_observation, reward, done = environment.step(action)
        next_state = _state_key(next_observation)
        learner.update(
            state,
            action,
            reward,
            next_state,
            list(ACTIONS),
            terminal=done,
        )
        observation = next_observation
        total_reward += reward
    return EpisodeResult(total_reward, environment.steps, environment.food_eaten)


def _state_key(observation: np.ndarray) -> str:
    energy, food_here, food_left, food_right = observation
    return (
        f"{int(energy < 0.3)}:"
        f"{int(food_here)}:{int(food_left)}:{int(food_right)}"
    )
