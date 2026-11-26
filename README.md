# B-RNA v0 software-only baseline and browser studio

**Status: Draft.** This repository packages a small, reproducible software experiment for a toy network that exchanges bounded latent vectors and uses learned gates. It is not a biological, hardware, actuator, or network-connected system.

## Scope and status

The evidence currently available is limited to a toy backbone and one synthetic task, three seeds, unequal parameter counts, and untuned training. It does not establish an advantage, generalization, or fitness for operational use. The no-communication result near chance is a valid result, not a failure to disclose.

## Browser studio

An installable, local-first browser studio is in [`web/`](web/). BRNA message-passing networks are the default, with configurable nodes, per-node features, hidden width, message-passing hops, readout node, in-degree, outputs, and communication mode. The default configuration follows the prototype dimensions and has 849 trainable parameters. Dense networks remain an optional model family. Both families support supervised training, loss-based evolutionary weight search, and contextual reward/policy updates. JSON model export/import, browser-local save, and taught-example conversation retrieval are included. This is a BRNA network-design workbench, not a game.

Run it locally from the repository root with `npm run serve:web`, then open `http://localhost:8000`. The app can be installed from a browser that supports PWAs and served over HTTPS; the service worker enables offline use after the first successful load. Browser data is stored in local storage and is not synchronized to the Python package.

The browser conversation feature retrieves explicitly taught examples; it is not a generative language model and does not update network weights. Policy learning uses contextual bandit rewards rather than multi-step state transitions. Evolutionary fitness is negative cross-entropy on the configured training examples; it does not represent a validated BRNA task fitness metric. The built-in XOR dataset is a smoke-test example, not real-world evidence.

Run the browser core tests with `npm run test:web`. The existing Python package and its agent-learning primitives remain available separately. See [the integration architecture](docs/integration_architecture.md) for boundaries and future work.

The Forge browser editor/runtime and the PhysX archive have not been integrated into the studio. The Evolution Unity source is not copied into this repository; its stated terms prohibit redistribution. The clean-room evolution optimizer is independent of that game.

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
