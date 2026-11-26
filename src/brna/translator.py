"""Software-only toy translator for local loop-control tests."""

from __future__ import annotations


class FastTranslator:
    def latency(self, text: str) -> float:
        return min(1.0, 0.001 * max(1, len(text)))

    def encode(self, text: str) -> list[int]:
        words = set(text.lower().split())
        value = 1 if "bright" in words else -1 if "dark" in words else 0
        return [value, -value] if value else []
