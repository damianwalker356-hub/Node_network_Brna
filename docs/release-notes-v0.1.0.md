# BRNA Studio v0.1.0

Initial published release of the software-only B-RNA research prototype and local-first studio.

## Highlights

- Train and evaluate BRNA message-passing and dense networks in the browser.
- Train and run a compact character n-gram model in-browser without a separate server.
- Optionally run the Python Studio for a small recurrent language model, local document ingestion, persistent role-based agents, and cooperative agent training.
- Add bounded latent-graph messaging, differentiable gates, episodic reward learning, and seeded Reasoning Gym tasks.
- Provide an opt-in Hugging Face backend with per-agent LoRA adapters and graph-conditioned policy experiments. It uses local model files only and does not download weights.
- Include a software-only closed-loop simulator, reproducible experiment tooling, tests, and traceability records.

## Install

```bash
python -m pip install -e '.[reasoning]'
brna-studio
```

For the optional Hugging Face backend, install `.[llm]` and pass `--backbone-dir` for a compatible model that is already on disk. Optional adapters include `--backbone-device auto`, `cpu`, and `cuda`.

## Limitations

This is experimental research software, not a general-purpose language model or a validated operational system. The browser generator is a compact character n-gram model; the Python model is a small from-scratch RNN. The Hugging Face adapter and graph-conditioned training have not been validated against a real checkpoint in this environment. The available experiment evidence is a toy backbone and synthetic task, with limited seeds, unequal parameter counts, and untuned training. No real-world advantage is established.
