# Changelog

All notable changes are recorded here.

## Unreleased

## 0.1.0 - 2026-11-27
- Add a browser-local character n-gram model that trains and generates without a separate server.
- Add the self-hosted Studio service, local recurrent text model, bounded document ingestion, and persistent cooperative agent training.
- Add bounded latent-graph messaging, differentiable gates, episodic policy updates, and verifier-scored Reasoning Gym tasks.
- Add an optional local-only Hugging Face backend with per-agent LoRA adapters and experimental graph-conditioned policy optimization.
- Add the software-only closed-loop B-RNA simulator, test coverage, traceability, and verification records.
- Document experimental scope, local-data behavior, optional dependencies, and model limitations.

**Limitations:** This release is research software. The browser model is a compact character n-gram generator; the Python model is a small from-scratch RNN. The optional Hugging Face adapter has not been validated against a real checkpoint in this environment. No real-world or operational advantage is established.
