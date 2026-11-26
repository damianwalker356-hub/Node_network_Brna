# Interface Control Document

**Status: Draft.** The product is software-only. The Python service binds to loopback by default and exposes local JSON HTTP interfaces; no external model provider, hardware, actuator, or biological interface is defined.

## Agent network interface

| Method and path | Request | Response / behavior |
|---|---|---|
| `GET /api/agents` | None | Node summaries, shared-corpus character count, reward baseline and feedback count |
| `POST /api/agents/create` | `name`, `role`; optional `hidden_size` (8–128) and non-negative integer `seed` | Creates and persists one independent character-RNN node; at most 8 nodes |
| `POST /api/agents/train` | `text` or `source: "examples"`; optional `epochs` (1–50), `learning_rate` (0–1], `sequence_length` (8–256) | Trains all current nodes on the same corpus and reports per-node losses |
| `POST /api/agents/upload` | `filename`, base64 `content_base64`; optional `epochs` | Extracts supported TXT/Markdown/DOCX/PDF locally and trains every current node |
| `POST /api/agents/think` | `prompt`; optional `max_new_tokens` (1–512) | Returns each node's private and shared-pass messages and the last shared-pass response |
| `POST /api/agents/feedback` | `reward` (-1 to 1); optional `discount` (0–1) and `learning_rate` (0–0.1] | Applies discounted episodic REINFORCE to the most recent collective response; each trajectory is consumed once |
| `GET /api/agents/datasets` | None | Returns the curated procedural task catalog or HTTP 503 when the optional dependency is missing |
| `POST /api/agents/datasets/train` | `task_name` (`basic_arithmetic` or `leg_counting`), optional `size` (1–128), `seed` (0–2^32-1), `epochs` (1–50) | Generates seeded, answer-verified examples and trains every current node on question/answer text |
| `POST /api/agents/reasoning-episode` | `task_name`, optional `seed`, `index` (0–127), token/discount/learning-rate settings | Generates an item, runs collective inference, scores the response with the task verifier, maps score [0,1] to reward [-1,1], and applies the reward to that episode |
| `POST /api/agents/grpo-episode` | `task_name`, optional `seed` (0–2^32-1), `index` (0–127), `group_size` (2–8), `max_new_tokens` (1–512), `kl_beta` (0–1), `learning_rate` (0–0.001] | Samples candidate groups per agent, scores them with the deterministic Reasoning Gym verifier, applies group-relative advantages and frozen-base KL regularization, and persists the adapters and graph prefix |
| `GET /api/agents/graph` | None | Reports latent-graph dimensions, active edges, episode state, and reward baseline; requires at least one current agent |
| `POST /api/agents/graph/train` | `examples` (1–256 items, each with one text `contexts` entry per agent and integer `action` from 0 to 3); optional `epochs` (1–100) | Trains graph transforms and directed edge gates from labeled actions, then saves a graph checkpoint |
| `POST /api/agents/graph/episode/start` | `{}` | Starts a bounded graph-policy episode |
| `POST /api/agents/graph/step` | `prompt`; optional `max_new_tokens` (1–512) | Runs each trained agent through the bounded latent graph and returns agent messages and a sampled discrete action |
| `POST /api/agents/graph/reward` | `reward` (-1 to 1), optional `done` (boolean) | Records feedback for the latest selected action; on `done: true`, updates the graph policy and participating agents with discounted episodic REINFORCE and persists the graph checkpoint |

The shared pass broadcasts all private-pass outputs to every node, then shares each earlier shared-pass output with later nodes. Each node has separate weights; the corpus is shared and the models do not exchange raw parameter tensors. Run records contain settings, per-node seeds, aggregate results, and source hashes; prompts and generated content are excluded.

The service defaults to its character-level RNN. To opt into the pretrained backend, install `.[llm]` and run `brna-studio --backbone-dir /path/to/local/model --backbone-device auto`; `BRNA_BACKBONE_DIR` is an alternative path setting. Loading is local-only, and model weights are not downloaded by the application. In this mode one causal-LM base is shared, with separate adapters under `agent-network/adapters/`. `/api/chat` uses the bounded agent graph when user agents exist; the graph emits continuous soft-prefix embeddings prepended to each node's prompt embeddings. Training updates adapters using causal next-token loss on supplied text. `/api/agents/grpo-episode` uses small verifier-scored groups, group-relative advantages, and a frozen-base KL penalty. This experimental interface is not a stable external compatibility contract or a validated production GRPO implementation. Binding the service to a non-loopback host can expose user data and model operations to other machines; loopback is the default.

Reasoning Gym endpoints require the optional `reasoning-gym==0.1.25` dependency (`pip install -e '.[reasoning]'`). The local source installation supports this extra; GitHub Pages does not run the API. Desktop archives do not currently bundle this optional package.

## B-RNA evaluation command

The software-only closed-loop research harness is a local CLI rather than a physical-device API:

```bash
PYTHONPATH=src python -m brna.hybrid_experiment --seeds 3 --steps 100
```

It records paired results, source hashes, parameter budgets, and limitations under `results/hybrid_records/`. It does not accept electrical stimulation settings, connect to an MEA, or perform biological control.

The user-agent latent graph is also available through the loopback API above. `/api/agents/graph/train` expects one bounded text context per current agent for every labeled example. In an episode, callers start an episode, request a step, execute or evaluate the returned discrete action in their own software task, then submit its reward before requesting another step. Mark the terminal reward with `done: true`. Graph checkpoints are stored under the local `agent-network/graphs/` directory and are tied to the current agent IDs. The graph accepts bounded latent messages and discrete action labels/rewards; it does not execute a task environment or accept or emit physical device commands.
