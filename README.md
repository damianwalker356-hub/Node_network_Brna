# B-RNA v0 software-only baseline and browser studio

**Status: Draft.** This repository packages a small, reproducible software experiment for a toy network that exchanges bounded latent vectors and uses learned gates. It is not a biological, hardware, actuator, or network-connected system.

**[Open BRNA Studio in your browser](https://damianwalker356-hub.github.io/Node_network_Brna/web/)** · [Desktop downloads](https://github.com/damianwalker356-hub/Node_network_Brna/releases) · [Source repository](https://github.com/damianwalker356-hub/Node_network_Brna)

## Scope and status

The evidence currently available is limited to a toy backbone and one synthetic task, three seeds, unequal parameter counts, and untuned training. It does not establish an advantage, generalization, or fitness for operational use. The no-communication result near chance is a valid result, not a failure to disclose.

## Browser studio

An installable, local-first browser studio is in [`web/`](web/). BRNA message-passing networks are the default, with configurable nodes, per-node features, hidden width, message-passing hops, readout node, in-degree, outputs, and communication mode. Dense networks remain optional. Both families support supervised training, evolutionary weight search, and contextual reward updates. Model import/export and browser-local save are included.

The [GitHub Pages app](https://damianwalker356-hub.github.io/Node_network_Brna/web/) is static but can train and run a compact character n-gram generator directly in your browser; no separate server or install is required. Its model and training text stay in that browser profile. For the optional Python-backed version, install the project with Python 3.12 and run `brna-studio`; this starts a local server and opens the browser app. That server provides a separate character-level RNN, PDF/DOCX extraction, and persistent agent-network features. Both from-scratch models are experimental and are not comparable to a modern large language model.

The browser-only model accepts `.txt` and `.md` training text directly. With the optional Python server running, the app also accepts `.docx` and `.pdf` for on-device extraction and trains its Python model. Legacy `.doc` and RTF files are rejected explicitly. Uploaded data is not network-egressed by the application.

The local **Agent network** page creates up to eight separately trained role-based model nodes. Nodes train on a shared corpus, exchange private drafts and shared revisions, and can receive discounted episodic REINFORCE updates from a user reward. This uses the existing experimental character-level RNN; it is not frontier-scale language modeling, and no parity with ChatGPT or Claude Sonnet is claimed. Current shared-corpus limit is 50,000 characters. Node checkpoints and privacy-preserving run records (settings, seeds, hashes, aggregate results; no prompts or responses) are stored under the local app data directory.

Install and launch locally:

```bash
python -m pip install -e '.[reasoning]'
brna-studio
```

The page can generate seeded arithmetic and leg-counting examples, train every current node on them, or run one task against Reasoning Gym's answer verifier and apply the result as episodic reward. It does not download a static corpus. The GitHub Pages app cannot access this Python-hosted Reasoning Gym task generator.

## B-RNA closed-loop software experiment

The research target is a hybrid of a gated B-RNA node network, separate digital encoder/decoder translators, a recurrent reservoir, and a matched-control evaluation harness. The repository has no tissue or MEA, so the current implementation is an explicitly software-only simulation: it uses twelve role-seeded graph nodes (sensory, researcher, builder, synthesizer, critic, and router), abstract feature codes, a seeded noisy recurrent reservoir, and critic-bounded outer feedback. Training can adapt both translators. Fixed and adaptive latency/amplitude encodings are compared on the same synthetic paired-feature task. The silicon-only baseline has the same total parameter-slot budget; a fixed random-reservoir readout is also included.

For the user-created language agents, `brna.agent_graph` provides three connected layers: `BoundedLatentNode` turns each trained model's hidden context into an L2-bounded message; `DifferentiableGatedGraph` and `DifferentiableGraphGateTrainer` learn directed message transforms and gates; and `ClosedLoopRLPolicyOptimizer` trains graph-selected actions from discounted episode rewards. `AgentGraphBridge` runs actual `AgentNetwork` nodes through that path and persists updated agent models. Graph checkpoints support save/load.

The local studio exposes graph status, supervised training, and caller-driven RL episodes through `/api/agents/graph` and `/api/agents/graph/*`. Start an episode, request a step, evaluate its discrete action in your own software task, and submit a reward; mark the final reward with `done: true`. Graph checkpoints are stored with the local agent data and tied to the current node IDs. This API does not execute an environment or connect to a physical B-RNA device.

Run a reproducible small comparison with:

```bash
PYTHONPATH=src python -m brna.hybrid_experiment --seeds 3 --steps 100
```

Records are written to `results/hybrid_records/`. Role labels seed distinct node embeddings; they do not guarantee semantic expertise. Equal parameter slots, training samples, batch indices, and update counts do not equalize compute time or energy. This simulation can test translator and network code paths, not whether living neurons add value: hypothesis H3 remains blocked until there is approved, measured culture data and a real hardware adapter. No electrical stimulation settings or device commands are generated.

For browser UI work without the model server, use `npm run serve:web`. Desktop ZIPs for Windows, macOS, and Linux are built by [the desktop release workflow](.github/workflows/desktop-releases.yml) when a `v*` version tag is published; the in-app **Desktop downloads** link opens the Releases page. Browser/PWA data and the local server’s model data are separate. See [the integration architecture](docs/integration_architecture.md) for limitations and interfaces.

Policy learning uses contextual bandit rewards rather than multi-step state transitions. Evolutionary fitness is negative cross-entropy on configured training examples; it is not a validated BRNA task fitness metric. The built-in XOR dataset is a smoke test, not real-world evidence.

Run browser tests with `npm run test:web`; install `.[dev,reasoning]` and run Python tests with `pytest`.

The Forge browser editor/runtime and the PhysX archive have not been integrated into the studio. The Evolution Unity source is not copied into this repository; its stated terms prohibit redistribution. The clean-room evolution optimizer is independent of that game.

The checkout originally supplied contains `brna_proto.py`, `run_record.json`, and a prototype verification PDF. The Rev B source, runner, test-results JSON, baseline PDF, compliance workbook, full NPR document, and figure described in project context were not in the repository tree observed at implementation time. The historical record is retained as received in `results/run_records/20261126T002255Z-source.json`; report values are aggregated from that record. Its legacy schema predates required git-commit/settings metadata.

## Install and test (Python 3.12)

```bash
python -m pip install -e '.[dev,reasoning]'
pytest
```

Long synthetic training regression is marked `slow` and excluded by default. Run it explicitly with `pytest -m slow`.

## Reproduce numbers and create a new record

Regenerate report from checked-in records: `python -m brna.reporting --results results/run_records --out docs/verification_report.md`. No report table is hand-entered. To make a fresh run (default comparison uses 10 seeds): `python -m brna.experiment --seeds 10`. `--equal-parameters` restricts variants to vector and no-communication models; it does not claim matching parameter counts unless verified in the resulting record. New records use a unique UTC run ID and refuse overwrite. New training is comparatively long; use `--steps` to set steps explicitly.

The historical aggregate table uses all results in the preserved source record, seeds 0, 1, and 2, 1,200 steps, batch size 256, and 5,000 test examples.

## Configuration and governance

Dependencies are pinned in `pyproject.toml`; dev pins are also in `requirements-dev.txt`. Use conventional commits. Protect `main` through repository settings (require PR review, status checks, and disallow force pushes); settings are not applied by this repository content change. See `GATE.md` and `HANDOFF.md`. No release tag is created here.

Proposed Class E software classification is **unconfirmed** pending verification against NPR 7150.2D Appendix D. This repo does not make that classification determination.
