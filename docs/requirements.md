# Requirements and traceability

**Status: Draft.** Requirement IDs in `traceability.csv` are project-level implementation identifiers, not a claim of NPR compliance. The Rev B T0–T9 cases are specified from the supplied description; the Rev B source itself was not present in the repository tree inspected for this work.

Every test declares its requirement ID using both `@pytest.mark.req('ID')` and its test docstring; the traceability test verifies mapping against `compliance/traceability.csv`.

## Agent-network and upload requirements

- **R-UPLOAD-01 shall** accept `.txt`, `.md`, `.docx`, and `.pdf` training uploads, extract bounded UTF-8 text, train the local language model, and reject unsupported or malformed files with a measured error.
- **R-AGENT-NET-01 shall** update sampled completion likelihood upward for positive episode advantage and downward for negative advantage.
- **R-AGENT-NET-02 shall** allow a bounded group of independently parameterized role agents to share training data and outputs, then update each node from discounted episode feedback.
- **R-AGENT-NET-03 shall** expose persistent agent creation, shared training, collective inference, and reward feedback through the loopback API.
- **R-AGENT-NET-04 shall** extract supported training documents locally and train every active model node on the extracted corpus.
- **R-REASONING-GYM-01 shall** generate reproducible examples from the curated Reasoning Gym tasks and verify each generated reference answer with the upstream verifier.
- **R-REASONING-GYM-02 shall** train each active model node on the selected generated task examples and support an end-to-end verifier-scored RL episode.
- **R-REASONING-GYM-03 shall** expose the curated task catalog, seeded dataset training, and verifier-scored episodes through the local service API.
- **R-HYBRID-01 shall** connect role-seeded B-RNA graph nodes to distinct abstract encoder/decoder transforms and a simulated recurrent reservoir, with critic-bounded outer feedback.
- **R-HYBRID-02 shall** compare adaptive translators against fixed latency/amplitude code baselines and numerically validate gradients through the software signal path.
- **R-HYBRID-03 shall** evaluate paired seeded software controls and explicitly mark physical-culture hypothesis H3 as unevaluable without experimental data.
- **R-HYBRID-04 shall** compare the software hybrid with a silicon-only baseline having the exact same total parameter-slot budget and matched task samples, minibatches, and update count.
- **R-AGENT-GRAPH-01 shall** wrap each trained user-created language agent as a role-aware node that emits bounded latent messages and consumes bounded peer messages.
- **R-AGENT-GRAPH-02 shall** train a directed, differentiable graph's node transforms and edge gates from supervised action labels, with numerically checked gate gradients and checkpoint persistence.
- **R-AGENT-GRAPH-03 shall** sample graph policy actions and update the policy from bounded rewards using discounted episodic REINFORCE.
- **R-AGENT-GRAPH-04 shall** expose supervised graph training and caller-driven episode start, action selection, and reward feedback through the loopback API, persisting graph checkpoints for the current agent set.
