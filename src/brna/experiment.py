"""CLI for generating reproducible B-RNA v0 run records."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from brna.proto import run


def repository_head() -> str | None:
    git_dir = Path(".git")
    head = git_dir / "HEAD"
    if not head.exists():
        return None
    text = head.read_text().strip()
    if text.startswith("ref:"):
        ref = git_dir / text.split(":", 1)[1].strip()
        if ref.exists():
            return ref.read_text().strip() or None
        packed = git_dir / "packed-refs"
        if packed.exists():
            name = text.split(":", 1)[1].strip()
            for line in packed.read_text().splitlines():
                if line and not line.startswith("#") and line.endswith(" " + name):
                    return line.split()[0]
        return None
    return text or None


def code_sha256() -> dict[str, str]:
    output = {}
    for path in (Path("brna_proto.py"), Path("src/brna/proto.py")):
        if path.exists():
            output[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return output


def build_record(
    seeds,
    steps=1500,
    batch=256,
    test_samples=5000,
    variants=None,
):
    variants = variants or ["vector", "nocomm", "quantized", "single_small", "single_large"]
    results = [
        run(kind, seed, steps=steps, batch=batch, test_samples=test_samples)
        for kind in variants
        for seed in seeds
    ]
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "timestamp": timestamp,
        "git_commit": repository_head(),
        "code_sha256": code_sha256(),
        "settings": {
            "steps": steps,
            "batch": batch,
            "test_samples": test_samples,
            "seeds": list(seeds),
            "variants": list(variants),
        },
        "seeds": list(seeds),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--equal-parameters", action="store_true")
    args = parser.parse_args(argv)
    variants = ["vector", "nocomm"] if args.equal_parameters else None
    record = build_record(
        range(args.seeds),
        steps=args.steps,
        batch=args.batch,
        test_samples=args.test_samples,
        variants=variants,
    )
    timestamp = record["timestamp"].replace("-", "").replace(":", "").replace("+00:00", "Z")
    destination = Path("results/run_records") / f"{timestamp}-run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
