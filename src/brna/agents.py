"""Editable, declarative agent definitions and linear action selection."""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

_WEIGHT_LINE = re.compile(r"weight\s+([a-zA-Z][\w-]*)\s*:\s*(.+)")


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    rules: tuple[str, ...] = ()
    weights: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("agent name must not be empty")
        object.__setattr__(self, "rules", tuple(self.rules))
        normalized = {}
        for action, values in self.weights.items():
            if not re.fullmatch(r"[a-zA-Z][\w-]*", action):
                raise ValueError(f"invalid action name: {action!r}")
            vector = tuple(float(value) for value in values)
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError(f"weights for {action!r} must be a non-empty finite vector")
            normalized[action] = vector
        if normalized and len({len(vector) for vector in normalized.values()}) != 1:
            raise ValueError("all action weight vectors must have the same length")
        object.__setattr__(self, "weights", normalized)

    @classmethod
    def from_source(cls, source: str) -> AgentDefinition:
        name = None
        rules = []
        weights = {}
        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("agent "):
                if name is not None:
                    raise ValueError(f"line {line_number}: duplicate agent declaration")
                try:
                    parts = shlex.split(line)
                except ValueError as exc:
                    raise ValueError(f"line {line_number}: invalid agent declaration") from exc
                if len(parts) != 2 or parts[0] != "agent":
                    raise ValueError(f"line {line_number}: expected agent \"name\"")
                name = parts[1]
                continue
            if name is None:
                raise ValueError(f"line {line_number}: agent declaration must come first")
            if line.startswith("rule "):
                try:
                    parts = shlex.split(line)
                except ValueError as exc:
                    raise ValueError(f"line {line_number}: invalid rule declaration") from exc
                if len(parts) != 2 or parts[0] != "rule" or not parts[1].strip():
                    raise ValueError(f"line {line_number}: expected rule \"description\"")
                rules.append(parts[1])
                continue
            match = _WEIGHT_LINE.fullmatch(line)
            if match:
                action, raw_values = match.groups()
                if action in weights:
                    raise ValueError(f"line {line_number}: duplicate weights for {action!r}")
                try:
                    weights[action] = tuple(float(value.strip()) for value in raw_values.split(","))
                except ValueError as exc:
                    raise ValueError(f"line {line_number}: weights must be numbers") from exc
                continue
            raise ValueError(f"line {line_number}: unsupported agent statement")
        if name is None:
            raise ValueError('missing agent declaration; expected agent "name"')
        return cls(name=name, rules=tuple(rules), weights=weights)

    def to_source(self) -> str:
        lines = [f"agent {shlex.quote(self.name)}"]
        lines.extend(f"rule {shlex.quote(rule)}" for rule in self.rules)
        lines.extend(
            f"weight {action}: " + ", ".join(format(value, ".17g") for value in values)
            for action, values in sorted(self.weights.items())
        )
        return "\n".join(lines) + "\n"

    @property
    def feature_count(self) -> int:
        return len(next(iter(self.weights.values()))) if self.weights else 0

    def choose_action(self, features: Sequence[float], actions: Sequence[str]) -> str:
        vector = np.asarray(features, dtype=float)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("features must be a finite one-dimensional vector")
        if not actions:
            raise ValueError("at least one action is required")
        if not self.weights:
            raise ValueError("agent has no action weights")
        if vector.size != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features, received {vector.size}")
        unknown = set(actions) - self.weights.keys()
        if unknown:
            raise ValueError(f"agent has no weights for actions: {sorted(unknown)}")
        return max(actions, key=lambda action: float(np.dot(self.weights[action], vector)))
