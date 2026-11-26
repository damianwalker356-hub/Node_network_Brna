"""Reproducible software-only evaluation harness for the B-RNA loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from brna.hybrid import BudgetMLP, ClosedLoopBRNA, RandomReservoirReadout
from brna.proto import make_data

VARIANTS = (
    "hybrid_adaptive_latency",
    "hybrid_fixed_latency",
    "hybrid_adaptive_amplitude",
    "hybrid_fixed_amplitude",
    "silicon_equal_budget",
    "random_reservoir_control",
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "src/brna/hybrid.py",
    "src/brna/hybrid_experiment.py",
    "src/brna/network.py",
    "src/brna/proto.py",
)


def _validate_settings(
    *,
    seeds: int,
    steps: int,
    batch_size: int,
    train_samples: int,
    test_samples: int,
    outer_loops: int,
) -> None:
    integers = {
        "seeds": seeds,
        "steps": steps,
        "batch_size": batch_size,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "outer_loops": outer_loops,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integers.values()
    ):
        raise TypeError("experiment settings must be integers")
    if seeds < 1 or steps < 1 or batch_size < 1 or test_samples < 1:
        raise ValueError("seeds, steps, batch_size, and test_samples must be positive")
    if train_samples < batch_size:
        raise ValueError("train_samples must be at least batch_size")
    if not 1 <= outer_loops <= 4:
        raise ValueError("outer_loops must be from 1 to 4")


def _create_model(
    variant: str,
    *,
    seed: int,
    outer_loops: int,
    system_budget: int,
):
    rng = np.random.default_rng(seed)
    if variant.startswith("hybrid_"):
        code = "latency" if variant.endswith("latency") else "amplitude"
        adaptive = "adaptive" in variant
        return ClosedLoopBRNA(
            rng,
            code=code,
            adaptive_translators=adaptive,
            outer_loops=outer_loops,
        )
    if variant == "silicon_equal_budget":
        return BudgetMLP(system_budget, rng)
    if variant == "random_reservoir_control":
        return RandomReservoirReadout(rng)
    raise ValueError(f"unknown variant: {variant}")


def _model_logits(model, inputs: np.ndarray) -> np.ndarray:
    if isinstance(model, ClosedLoopBRNA):
        return np.asarray(model.run(inputs)["logits"], dtype=float)
    return np.asarray(model.logits(inputs), dtype=float)


def _train_batch(model, inputs: np.ndarray, labels: np.ndarray) -> None:
    model.train_step(inputs, labels)


def _parameter_counts(model) -> tuple[int, int]:
    if isinstance(model, ClosedLoopBRNA):
        return model.parameter_count, model.system_parameter_count
    if isinstance(model, BudgetMLP):
        return model.parameter_count, model.parameter_count
    if isinstance(model, RandomReservoirReadout):
        return model.parameter_count, model.system_parameter_count
    raise TypeError(f"unsupported model type: {type(model).__name__}")


def _source_hashes() -> dict[str, str]:
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def build_record(
    *,
    seeds: int = 3,
    steps: int = 100,
    batch_size: int = 64,
    train_samples: int = 2048,
    test_samples: int = 512,
    outer_loops: int = 2,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    _validate_settings(
        seeds=seeds,
        steps=steps,
        batch_size=batch_size,
        train_samples=train_samples,
        test_samples=test_samples,
        outer_loops=outer_loops,
    )
    unknown = set(variants) - set(VARIANTS)
    if not variants or unknown:
        raise ValueError(f"variants must be selected from {', '.join(VARIANTS)}")

    reference = ClosedLoopBRNA(np.random.default_rng(0), outer_loops=outer_loops)
    system_budget = reference.system_parameter_count
    if reference.parameter_count <= 0 or system_budget <= reference.parameter_count:
        raise RuntimeError("computed B-RNA parameter budget is invalid")

    results = []
    for seed in range(seeds):
        train_inputs, train_labels = make_data(
            train_samples, np.random.default_rng(seed)
        )
        test_inputs, test_labels = make_data(
            test_samples, np.random.default_rng(10_000 + seed)
        )
        batch_rng = np.random.default_rng(20_000 + seed)
        batch_indices = batch_rng.integers(
            0, train_samples, size=(steps, batch_size)
        )
        for variant in variants:
            model = _create_model(
                variant,
                seed=30_000 + seed,
                outer_loops=outer_loops,
                system_budget=system_budget,
            )
            trainable_parameters, total_parameters = _parameter_counts(model)
            if variant == "silicon_equal_budget" and total_parameters != system_budget:
                raise RuntimeError("silicon baseline failed to match the system parameter budget")
            for indices in batch_indices:
                _train_batch(model, train_inputs[indices], train_labels[indices])
            logits = _model_logits(model, test_inputs)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            loss = float(
                np.mean(
                    np.logaddexp(0.0, logits)
                    - test_labels * logits
                )
            )
            results.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "test_accuracy": float(np.mean((logits > 0) == test_labels)),
                    "test_loss": loss,
                    "test_mean_probability": float(probabilities.mean()),
                    "train_updates": steps,
                    "train_samples": train_samples,
                    "batch_size": batch_size,
                    "trainable_parameters": trainable_parameters,
                    "total_system_parameters": total_parameters,
                    "outer_loops": outer_loops if variant.startswith("hybrid_") else 1,
                    "translator_code": (
                        "latency"
                        if variant.endswith("latency")
                        else "amplitude"
                        if variant.endswith("amplitude")
                        else "not_applicable"
                    ),
                    "adaptive_translators": "adaptive" in variant,
                    "node_roles": (
                        list(model.node_roles)
                        if isinstance(model, ClosedLoopBRNA)
                        else []
                    ),
                }
            )

    grouped = {
        variant: [row for row in results if row["variant"] == variant]
        for variant in variants
    }
    summary = {
        variant: {
            "mean_test_accuracy": float(
                np.mean([row["test_accuracy"] for row in rows])
            ),
            "std_test_accuracy": float(
                np.std([row["test_accuracy"] for row in rows])
            ),
            "mean_test_loss": float(np.mean([row["test_loss"] for row in rows])),
            "seeds": [row["seed"] for row in rows],
        }
        for variant, rows in grouped.items()
    }
    by_variant_seed = {
        (row["variant"], row["seed"]): row["test_accuracy"] for row in results
    }

    def paired_delta(left: str, right: str) -> dict[str, Any] | None:
        if left not in grouped or right not in grouped:
            return None
        differences = [
            by_variant_seed[(left, seed)] - by_variant_seed[(right, seed)]
            for seed in range(seeds)
        ]
        return {
            "left": left,
            "right": right,
            "per_seed_accuracy_difference": differences,
            "mean_accuracy_difference": float(np.mean(differences)),
        }

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "timestamp": timestamp,
        "git_commit": _repository_head(),
        "source_sha256": _source_hashes(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "software_only": True,
        "task": {
            "name": "paired_feature_parity",
            "source": "synthetic task from brna.proto.make_data",
            "train_samples": train_samples,
            "test_samples": test_samples,
        },
        "budget_definition": {
            "equalized": "total parameter slots, training samples, batch indices, and update count for silicon_equal_budget vs hybrid variants",
            "silicon_total_parameter_budget": system_budget,
            "physical_energy_or_wall_time_matched": False,
        },
        "settings": {
            "seeds": list(range(seeds)),
            "steps": steps,
            "batch_size": batch_size,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "outer_loops": outer_loops,
            "variants": list(variants),
            "node_roles": list(reference.node_roles),
        },
        "comparisons": {
            "H1_adaptive_vs_fixed_latency": paired_delta(
                "hybrid_adaptive_latency", "hybrid_fixed_latency"
            ),
            "H1_adaptive_vs_fixed_amplitude": paired_delta(
                "hybrid_adaptive_amplitude", "hybrid_fixed_amplitude"
            ),
            "H2_hybrid_vs_silicon_equal_parameter_budget": paired_delta(
                "hybrid_adaptive_latency", "silicon_equal_budget"
            ),
            "H3_culture_contribution": {
                "status": "not_evaluable_without physical culture data",
                "digital_control": "random_reservoir_control is a software-only recurrent reservoir baseline, not a cultured-neuron condition",
            },
        },
        "summary": summary,
        "results": results,
        "limitations": [
            "The simulated reservoir is a seeded numerical dynamical system, not a neuronal culture or MEA.",
            "Synthetic paired-feature parity is a software smoke task, not spoken-digit recognition or a biological benchmark.",
            "Equal parameter slots, examples, batches, and gradient updates do not imply equal FLOPs, wall-clock time, energy, or physical budgets.",
            "The random-reservoir control cannot establish that a living culture contributes beyond a silicon reservoir; H3 requires approved experimental data.",
            "Outer-loop training uses truncated feedback: gradients update the translators and node network within each round but do not backpropagate through earlier outer-loop feedback.",
            "No stimulation voltages, pulse timings, hardware commands, animal procedures, or biological claims are implemented.",
        ],
    }


def _repository_head() -> str | None:
    git_directory = ROOT / ".git"
    head = git_directory / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref_name = text.split(":", 1)[1].strip()
        ref = git_directory / ref_name
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip() or None
        packed_refs = git_directory / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and line.endswith(f" {ref_name}"):
                    return line.split()[0]
        return None
    return text or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--outer-loops", type=int, default=2)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    record = build_record(
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        outer_loops=args.outer_loops,
    )
    destination_directory = args.output_directory or ROOT / "results" / "hybrid_records"
    destination_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_directory / f"{run_id}-hybrid-evaluation.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
