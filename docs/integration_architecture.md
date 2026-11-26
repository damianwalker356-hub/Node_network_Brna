# Combined agent-learning direction

**Status: Draft.** This is an incremental integration of the requested capabilities into the existing Python package. It does not claim that a finished application or a real conversational language model is present.

## Initial Python core

- **Agent builder:** `brna.agents.AgentDefinition` stores an agent name, descriptive rules, and action-weight vectors. Its small text format is editable and validated before use.
- **Conversational learning:** `brna.conversation.ConversationMemory` stores explicit prompt/response examples and retrieves the closest taught example by token overlap. It does not invent an answer or imply language-model generalization.
- **Reinforcement learning:** `brna.reinforcement.QLearner` provides an epsilon-greedy discrete-action Q-table and a configurable temporal-difference update.
- **Evolution-style learning:** `brna.evolution.EvolutionTrainer` creates and evolves numerical policy genomes with elitism, crossover, mutation, and seeded reproducibility.
- **Creature task:** `brna.creatures.CreatureEnvironment` is an original, bounded foraging environment. It can run either editable weighted policies or a Q-learner and returns episode reward, length, and food collected.
- **Existing B-RNA prototype:** `brna.proto` remains the current neural-network experiment, separate from the new symbolic-memory and policy-learning modules.

## Authoring and execution boundary

Agent text is parsed into data; no `eval`, `exec`, shell, network, or arbitrary user-code execution is provided. The Forge repository's Python-like language and browser editor are not yet integrated. Adding a code editor is a separate UI task; executing user code would need a deliberately isolated runtime and dedicated security tests.

## Source and compatibility notes

The Forge source repository is `damianwalker356-hub/forge-agent`. It is a browser application whose model, skills, and Forge-language editor are implemented in JavaScript and browser storage. The current target is a Python package, so this first step ports capability boundaries rather than copying the whole browser app.

The Evolution Unity repository identifies Keiwan Donyagard as copyright holder and permits personal/educational modification while disallowing relicensing or redistribution. Its Unity/C# code is not included here. The Python evolution module is independently implemented from general algorithmic concepts; obtain explicit compatible rights before redistributing any of the original game's source or builds.

## Next integration stages

1. Add persistence and a small CLI around agent definitions, conversation memory, and Q-tables.
2. Connect evolutionary genomes to a documented toy creature environment and measure episode rewards.
3. Design an interactive authoring/chat surface for this Python core.
4. Evaluate whether the Forge language can be used as a safe policy authoring format; do not treat its existing native runtime as a sandbox.
5. Add multi-agent experiments only after the single-agent data and evaluation contracts are stable.

All stages remain software-only and experimental. Performance claims and game equivalence require independent evidence.
