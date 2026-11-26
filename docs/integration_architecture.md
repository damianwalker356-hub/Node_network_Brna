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

The browser UI includes a compact character n-gram model that trains and generates in the current browser profile, including when served statically from GitHub Pages. It requires no local server, and its training corpus is kept in browser storage. The optional `brna-studio` Python command runs a service on `127.0.0.1:8765` and serves the same UI and JSON API from one origin. It stores a separate character-level recurrent language model and conversation memory in the user data directory; it also enables document extraction and agent-network endpoints.

Without the Python service, browser training accepts `.txt` and `.md` text directly. The `/api/upload` endpoint accepts bounded base64-encoded `.txt`, `.md`, `.docx`, and `.pdf` files, extracts text locally, and trains the Python character model. Legacy `.doc` and RTF formats are explicitly rejected rather than guessed.

Both from-scratch models are small experimental character predictors, not pretrained or general-purpose large language models. They have no built-in knowledge, and useful output depends on the quality and amount of training text supplied by the user. Neither makes a claim of general language understanding or production readiness.

## Cooperative model-agent network

The local studio can create up to eight independently parameterized model nodes, each with a user-selected name and role. All nodes train on a shared corpus; the models retain separate weights and checkpoints. A collective turn has a private draft pass followed by a shared pass. In the shared pass each node sees all private drafts and shared-pass messages produced earlier in node order; the last node's revision is returned. This provides directed, bounded message exchange rather than simultaneous unrestricted inference.

The **Reinforce this episode** action applies a character-level sequence REINFORCE update to every generated message, using a discounted terminal user reward and an exponential running reward baseline. This is a genuine episodic policy-gradient update, but it is not PPO, RLHF, a learned value model, or a validated task policy. Its reward signal is the user's rating of the collective response. The update changes sampled response likelihood; it does not guarantee improved quality and can degrade other behavior.

The existing character model is an Elman RNN with Unicode character tokens, hidden width bounded to 8–128, and a shared corpus capped at 50,000 characters. It is not a modern transformer or a foundation model, and this implementation should not be described as close to ChatGPT or Claude Sonnet. The environment measured for the initial implementation had two CPU cores, less than 1 GiB available memory, and no detected NVIDIA GPU. Model quality beyond unit-test behavior remains unmeasured; larger-scale from-scratch training requires a separate compute and data program.

Node checkpoints, the shared training corpus, network metadata, and run records are stored in the local application data directory under `agent-network/`. Each creation, training, collective-inference, and reward-update run records a UTC timestamp, node seeds, settings, aggregate results, and SHA-256 hashes of the relevant source files. Prompts and generated text are intentionally excluded from run records. Checkpoints persist the NumPy random-generator state so later sampling can continue reproducibly.

The optional Reasoning Gym integration uses the pinned `reasoning-gym==0.1.25` Apache-2.0 dependency. It generates `basic_arithmetic` and `leg_counting` items at runtime with a recorded seed; generated question/answer examples are applied as supervised text training to every active node. A separate verified-episode route uses the task's `score_answer` method and maps its [0,1] score to a [-1,1] REINFORCE reward. The system does not download or commit a static corpus. Other Reasoning Gym tasks are not exposed because the upstream repository's notice identifies separate third-party licenses and data sources.

## B-RNA closed-loop software simulation

`brna.hybrid.ClosedLoopBRNA` composes the existing 12-node gated message-passing network with role-seeded node embeddings for sensory, researcher, builder, synthesizer, critic, and router roles; separate abstract `DigitalEncoder` and `DigitalDecoder` transforms; and a seeded, noisy recurrent reservoir. Role labels do not guarantee semantic expertise. Latency and amplitude code modes use normalized feature tensors, not electrical pulse specifications. The simulator returns reservoir features to the node network; a `Critic` limits outer feedback to at most four rounds. The translators and node network can be trained from task loss. Gradient propagation is verified numerically through the encoder, decoder, reservoir, and node-network input.

`brna.hybrid_experiment` runs paired synthetic task data against adaptive/fixed translator modes, an exact-total-parameter silicon-only MLP, and a fixed random-reservoir readout. Its default paired-feature parity task is a software check only. Total parameter slots, task samples, training batches, and update counts are matched for the principal silicon comparison; FLOPs, wall-clock time, energy, and any physical resource budget are not. H1/H2 measurements in this harness are software-only. H3 is deliberately marked unmeasured because no neuronal culture or MEA data exists. The simulator issues no device commands and does not specify stimulation parameters or biological procedures.

The separate text-agent integration is in `brna.agent_graph`: `BoundedLatentNode` deterministically projects each trained `CharacterLanguageModel` context into a bounded latent vector; `AgentGraphBridge` passes those vectors through a learned directed graph and back into each role model as peer context; `DifferentiableGraphGateTrainer` updates graph weights and gates from labeled actions; and `ClosedLoopRLPolicyOptimizer` applies discounted episodic REINFORCE to graph actions. The loopback studio exposes this workflow through `GET /api/agents/graph`, `POST /api/agents/graph/train`, and the episode start/step/reward routes documented in `docs/icd.md`. Graph checkpoints are saved under the local `agent-network/graphs/` directory and loaded for the same set of agent IDs. Actions are discrete outputs for a caller-provided software task; the service does not execute an environment or send latent vectors to the simulated reservoir automatically, and it has no physical-device interface.
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
5. Define task/environment-specific state, action, reward, episode termination, and holdout evaluation before claiming RL task performance; current language-agent feedback uses discounted terminal rewards.
6. Grow the from-scratch text model only with licensed, reviewed training data and measurable evaluation; do not infer general conversational quality from training loss.

All stages remain software-only and experimental. Performance claims and game equivalence require independent evidence.
