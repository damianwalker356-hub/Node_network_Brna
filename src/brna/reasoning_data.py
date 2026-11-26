"""Curated procedural reasoning tasks from the optional Reasoning Gym package."""

from __future__ import annotations

import math
from typing import Any

MAX_REASONING_EXAMPLES = 128
TASKS = {
    "basic_arithmetic": {
        "label": "Basic arithmetic",
        "description": "Seeded arithmetic expressions with verifier-scored answers.",
    },
    "leg_counting": {
        "label": "Leg counting",
        "description": "Seeded counting problems with verifier-scored answers.",
    },
}


def available_tasks() -> dict[str, dict[str, str]]:
    _reasoning_gym()
    return {name: dict(details) for name, details in TASKS.items()}


def generate_examples(
    task_name: str,
    *,
    size: int = 16,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], Any]:
    if not isinstance(task_name, str) or task_name not in TASKS:
        raise ValueError(f"task_name must be one of: {', '.join(TASKS)}")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 1 <= size <= MAX_REASONING_EXAMPLES
    ):
        raise ValueError(f"size must be a whole number from 1 to {MAX_REASONING_EXAMPLES}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a whole number from 0 to 4294967295")

    reasoning_gym = _reasoning_gym()
    if task_name == "basic_arithmetic":
        dataset = reasoning_gym.create_dataset(
            task_name,
            size=size,
            seed=seed,
            min_terms=2,
            max_terms=4,
            min_digits=1,
            max_digits=2,
            operators=["+", "-", "*"],
            allow_parentheses=False,
            allow_negation=False,
            format_style="simple",
        )
    else:
        dataset = reasoning_gym.create_dataset(
            task_name,
            size=size,
            seed=seed,
            min_animals=2,
            max_animals=4,
            min_instances=1,
            max_instances=4,
        )

    examples = []
    for index in range(size):
        entry = dataset[index]
        question, answer = entry.get("question"), entry.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError(f"Reasoning Gym task {task_name!r} returned an invalid example")
        if dataset.score_answer(answer=answer, entry=entry) != 1.0:
            raise RuntimeError(f"Reasoning Gym task {task_name!r} failed its answer verifier")
        examples.append(
            {
                "question": question,
                "answer": answer,
                "metadata": entry.get("metadata", {}),
            }
        )
    return examples, dataset


def score_answer(task_name: str, answer: str, *, seed: int = 0, index: int = 0) -> float:
    examples, dataset = generate_examples(task_name, size=index + 1, seed=seed)
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    score = dataset.score_answer(answer=answer, entry=examples[index])
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise RuntimeError("Reasoning Gym returned an invalid answer score")
    return float(max(0.0, min(1.0, score)))


def _reasoning_gym():
    try:
        import reasoning_gym
    except ImportError as exc:
        raise RuntimeError(
            "Reasoning Gym support is optional; install this project with "
            "`pip install -e '.[reasoning]'`"
        ) from exc
    return reasoning_gym
