# Verification and validation plan

**Status: Draft.** Unit tests cover toy bounds, controls, determinism, translator behavior, gradients, metadata, traceability, no-egress policy, report completeness and an explicitly slow synthetic regression. These tests are software checks, not independent validation, NPR compliance evidence, or validation of a real backbone. Results are reported only from checked-in run records.

Upload corpus extraction is tested for UTF-8 text/Markdown, DOCX, and a generated PDF fixture. PDF extraction uses the pinned `pypdf` package.

## Earlier measured verification run - 2026-11-26

The following pre-agent-network results are from `results/vv/20261126T044237Z-verification.json`:

- Git commit: `a2ab4d84d48efbe02090f414016505caa1c2438e`
- Worktree: dirty; this evidence describes the implementation before an additional commit.
- Python: 3.14.2; NumPy 2.5.5; pypdf 5.1.0; pytest 9.2.0.
- Python verification: 35 passed, 1 slow test deselected.
- Node.js: v24.22.2.
- Browser verification: 9 passed, 0 failed.
- Raw Python output: `results/vv/python-20261126.txt`.
- Raw browser output: `results/vv/web-20261126.txt`.

The active environment did not match the pinned Python package versions in `pyproject.toml` for NumPy and pytest. The record reports the measured environment and does not claim an equivalent pinned-environment run.

## Cooperative agent network verification

The current implementation adds independent model checkpoints, shared corpus training, private/shared two-pass communication, persistent network metadata, and discounted user-reward REINFORCE updates. Focused tests verify likelihood movement under positive/negative advantages, message exchange, persistence, and the local API lifecycle. The synthetic training fixtures are not a language-quality benchmark. No capability comparison against any hosted model was performed.

The measured implementation run is recorded in `results/vv/20261126T051255Z-agent-network-verification.json`:

- Python: 40 passed, 1 slow test deselected; run by itself after an initial concurrent run was terminated under load.
- Browser: 9 passed, 0 failed. JavaScript syntax, Ruff, and `git diff --check` passed.
- Runtime: Python 3.14.2, NumPy 2.5.5, pytest 9.2.0, Node.js v24.22.2.
- Measured resources: 2 available CPU cores, 832 MiB available memory at the check, and no NVIDIA GPU detected.
- The project declares Python 3.12 and NumPy 2.2.3 / pytest 8.3.5 pins; those declared versions were not used for this run.
- Model language quality and RL task performance remain unmeasured. The run verifies software behavior only and does not establish that the model is close to frontier systems.
- Raw outputs: `results/vv/20261126T051000Z-agent-network-python.txt` (terminated concurrent attempt), `results/vv/20261126T051255Z-agent-network-python.txt` (successful isolated rerun), `results/vv/20261126T051308Z-agent-network-web.txt`, and `results/vv/20261126T051308Z-agent-network-lint.txt`.

## Curated Reasoning Gym verification

The current implementation adds optional, seeded generation of `basic_arithmetic` and `leg_counting` examples, per-node supervised training, and task-verifier-scored episodic reward updates through the local service/UI. This is a narrow procedural task integration, not a general language-capability benchmark.

The measured run is recorded in `results/vv/20261126T053500Z-reasoning-gym-verification.json`:

- Python: 43 passed, 1 explicitly slow test deselected.
- Browser: 9 passed, 0 failed. Ruff, JavaScript syntax, and `git diff --check` passed.
- Runtime: Python 3.14.2, NumPy 2.5.5, pytest 9.2.0, Ruff 0.17.6, Node.js v24.22.2.
- The optional Reasoning Gym distribution reports 0.1.25; its module reports 0.1.19. Only the two reviewed tasks above are exposed.
- The project declares Python 3.12 and NumPy 2.2.3 / pytest 8.3.5 pins; those declared versions were not used for this run.
- Model language quality, task-level RL performance, and frontier-model parity remain unmeasured.
- Raw outputs: `results/vv/20261126T053500Z-reasoning-gym-python.txt`, `results/vv/20261126T053500Z-reasoning-gym-web.txt`, and `results/vv/20261126T053500Z-reasoning-gym-checks.txt`.
