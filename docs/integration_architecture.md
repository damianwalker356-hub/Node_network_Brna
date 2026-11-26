# Combined agent-learning direction

**Status: Draft.** The project now includes a browser-first, installable PWA alongside its Python experiment package. It is a neural-network design and training workbench for BRNA research, not a game. It does not claim a production-ready BRNA or generative language model.

## Browser studio

The static PWA in `web/` provides the browser interface:

- Design the prototype-shaped message-passing network with configurable nodes, node features, hidden width, hops, readout node, incoming edges, output count, and communication mode; inspect its directed graph and parameter count. The default has 12 nodes, 4 features per node, width 16, 3 hops, and 849 parameters.
- Optionally create a dense network with an input layer, up to six hidden layers, and an output layer.
- Create/switch between locally saved model definitions; export/import an individual model as JSON.
- Train either network family with backpropagation on small, validated numeric datasets.
- Run evolutionary search over network-weight candidates, with deterministic seeds and negative cross-entropy as the explicitly defined example fitness function.
- Apply contextual policy-gradient updates from user-provided action rewards. This is a bandit-style update; delayed rewards and state transitions are not implemented.
- Teach prompt/response pairs into local conversation memory and retrieve a sufficiently similar example.
- Install and use the UI offline after a successful initial load from localhost or HTTPS.

The `brna-studio` Python command runs the self-hosted service on `127.0.0.1:8765` and serves the same UI and JSON API from one origin. It stores a character-level recurrent language model and conversation memory in the user data directory. The model is implemented from scratch and can be trained from taught pairs or an imported plain-text corpus. The GitHub Pages version remains static; generation requires running the local service or a desktop release.

The from-scratch model is a small experimental character predictor, not a pretrained or general-purpose large language model. It has no built-in knowledge, and useful output depends on the quality and amount of training text supplied by the user. This implementation deliberately makes no claim of general language understanding or production readiness.
- Install and use offline as a PWA after a successful initial load from localhost or HTTPS.

The built-in examples validate basic supervised training only. Fitness and policy rewards are not validated against real BRNA tasks. Browser-stored neural-network models remain separate from the Python package. Agent text is parsed as declarative data; the application does not execute arbitrary user-authored code.

## Initial Python core

- **Agent builder:** `brna.agents.AgentDefinition` stores an agent name, descriptive rules, and action-weight vectors. Its small text format is editable and validated before use.
- **Conversational learning:** `brna.conversation.ConversationMemory` stores explicit prompt/response examples and retrieves the closest taught example by token overlap. It does not invent an answer or imply language-model generalization.
- **Reinforcement learning:** `brna.reinforcement.QLearner` provides an epsilon-greedy discrete-action Q-table and a configurable temporal-difference update.
- **Evolution-style learning:** `brna.evolution.EvolutionTrainer` creates and evolves numerical policy genomes with elitism, crossover, mutation, and seeded reproducibility.
- **Creature task (Python experiment only):** `brna.creatures.CreatureEnvironment` is an original bounded foraging environment for policy experiments. It is not the browser product or a game integration.
- **Existing B-RNA prototype:** `brna.proto` remains the current neural-network experiment, separate from the new symbolic-memory and policy-learning modules.

## Authoring and execution boundary

Agent text is parsed into data; no `eval`, `exec`, shell, network, or arbitrary user-code execution is provided. The Forge repository's Python-like language and browser editor are not yet integrated. Adding a code editor is a separate UI task; executing user code would need a deliberately isolated runtime and dedicated security tests.

## Source and compatibility notes

The Forge source repository is `damianwalker356-hub/forge-agent`. It is a browser application whose model, skills, and Forge-language editor are implemented in JavaScript and browser storage. The current target is a Python package, so this first step ports capability boundaries rather than copying the whole browser app.

The Evolution Unity repository identifies Keiwan Donyagard as copyright holder and permits personal/educational modification while disallowing relicensing or redistribution. Its Unity/C# code is not included here. The Python evolution module is independently implemented from general algorithmic concepts; obtain explicit compatible rights before redistributing any of the original game's source or builds.

## Next integration stages

1. Define and review representative BRNA task datasets, evaluation splits, and task-specific fitness functions.
2. Connect the browser model format to the Python experiments with versioned import/export and compatibility tests.
3. Add editable datasets and architecture-aware training settings, with validation and reproducible run metadata.
4. Evaluate whether the Forge language can be used as a safe policy-authoring format; do not treat its existing native runtime as a sandbox.
5. Add multi-step reinforcement learning only after state/action/reward and evaluation contracts are defined.
6. Grow the from-scratch text model only with licensed, reviewed training data and measurable evaluation; do not infer general conversational quality from training loss.

All stages remain software-only and experimental. Performance claims and game equivalence require independent evidence.
