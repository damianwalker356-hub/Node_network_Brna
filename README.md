# B-RNA v0 software-only baseline

**Status: Draft.** This repository packages a small, reproducible software experiment for a toy network that exchanges bounded latent vectors and uses learned gates. It is not a biological, hardware, actuator, or network-connected system.

## Scope and status

The evidence currently available is limited to a toy backbone and one synthetic task, three seeds, unequal parameter counts, and untuned training. It does not establish an advantage, generalization, or fitness for operational use. The no-communication result near chance is a valid result, not a failure to disclose.

The checkout originally supplied contains `brna_proto.py`, `run_record.json`, and a prototype verification PDF. The Rev B source, runner, test-results JSON, baseline PDF, compliance workbook, full NPR document, and figure described in project context were not in the repository tree observed at implementation time. The historical record is retained as received in `results/run_records/20261126T002255Z-source.json`; report values are aggregated from that record. Its legacy schema predates required git-commit/settings metadata.

## Install and test (Python 3.12)

```bash
python -m pip install -e '.[dev]'
pytest
```

Long synthetic training regression is marked `slow` and excluded by default. Run it explicitly with `pytest -m slow`.

## Reproduce numbers and create a new record

Regenerate report from checked-in records: `python -m brna.reporting --results results/run_records --out docs/verification_report.md`. No report table is hand-entered. To make a fresh run (default comparison uses 10 seeds): `python -m brna.experiment --seeds 10`. `--equal-parameters` restricts variants to vector and no-communication models; it does not claim matching parameter counts unless verified in the resulting record. New records use a unique UTC run ID and refuse overwrite. New training is comparatively long; use `--steps` to set steps explicitly.

The historical aggregate table uses all results in the preserved source record, seeds 0, 1, and 2, 1,200 steps, batch size 256, and 5,000 test examples.

## Configuration and governance

Dependencies are pinned in `pyproject.toml`; dev pins are also in `requirements-dev.txt`. Use conventional commits. Protect `main` through repository settings (require PR review, status checks, and disallow force pushes); settings are not applied by this repository content change. See `GATE.md` and `HANDOFF.md`. No release tag is created here.

Proposed Class E software classification is **unconfirmed** pending verification against NPR 7150.2D Appendix D. This repo does not make that classification determination.
