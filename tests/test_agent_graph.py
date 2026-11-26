import numpy as np
import pytest

from brna.agent_graph import (
    AgentGraphBridge,
    ClosedLoopRLPolicyOptimizer,
    DifferentiableGatedGraph,
    DifferentiableGraphGateTrainer,
)
from brna.agent_network import AgentNetwork


def _trained_agent_network(directory):
    agents = AgentNetwork(directory, seed=40)
    agents.create_node("Sensor", "Extract relevant signals.", hidden_size=8, seed=1)
    agents.create_node("Critic", "Review the other node.", hidden_size=8, seed=2)
    agents.train_all(
        "User: signal\nAssistant: detect a stable pattern.\n" * 4,
        epochs=2,
        sequence_length=32,
    )
    return agents


@pytest.mark.req("R-AGENT-GRAPH-01")
def test_user_language_agents_emit_bounded_role_conditioned_latents(tmp_path):
    """R-AGENT-GRAPH-01: adapt trained user agents to bounded latent vectors."""
    agents = _trained_agent_network(tmp_path)
    wrappers = agents.bounded_latent_nodes(latent_size=8, max_norm=0.75)
    first = wrappers[0].encode_text("detect a pattern")
    repeated = wrappers[0].encode_text("detect a pattern")
    message = wrappers[1].respond(
        "review this signal",
        wrappers[1].encode_text("peer context"),
        max_new_tokens=8,
    )

    assert first.shape == (8,)
    assert np.linalg.norm(first) <= 0.75 + 1e-8
    np.testing.assert_array_equal(first, repeated)
    assert message.role == "Review the other node."
    assert np.linalg.norm(message.latent) <= 0.75 + 1e-8


@pytest.mark.req("R-AGENT-GRAPH-02")
def test_differentiable_graph_trainer_updates_edge_gates_and_persists(tmp_path):
    """R-AGENT-GRAPH-02: train gated directed links from supervised task labels."""
    rng = np.random.default_rng(42)
    latents = rng.normal(size=(16, 3, 4))
    latents /= np.maximum(1.0, np.linalg.norm(latents, axis=-1, keepdims=True))
    actions = np.asarray([int(row[0, 0] > 0) for row in latents])
    graph = DifferentiableGatedGraph(
        node_count=3,
        latent_size=4,
        action_count=2,
        hidden_size=4,
        hops=2,
        seed=43,
    )
    trainer = DifferentiableGraphGateTrainer(graph)
    initial_gates = graph.edge_gates.copy()
    report = trainer.fit(latents, actions, epochs=3)

    assert report["updates"] == 3
    assert all(np.isfinite(report["losses"]))
    assert not np.array_equal(graph.edge_gates, initial_gates)
    expected = graph.forward(latents).action_logits
    graph_path = tmp_path / "gated-graph.npz"
    graph.save(graph_path)
    restored = DifferentiableGatedGraph.load(graph_path)
    np.testing.assert_array_equal(restored.forward(latents).action_logits, expected)


@pytest.mark.req("R-AGENT-GRAPH-02")
def test_differentiable_gate_gradient_matches_finite_differences():
    """R-AGENT-GRAPH-02: check the learned edge-gate derivative numerically."""
    rng = np.random.default_rng(48)
    latents = rng.normal(size=(2, 3, 4))
    latents /= np.maximum(1.0, np.linalg.norm(latents, axis=-1, keepdims=True))
    actions = np.asarray([0, 1])
    graph = DifferentiableGatedGraph(
        node_count=3,
        latent_size=4,
        action_count=2,
        hidden_size=4,
        hops=2,
        seed=49,
    )
    index = graph.edge_gates.size - 1
    epsilon = 1e-6

    def loss():
        logits = graph.forward(latents).action_logits
        probabilities = np.stack(
            [
                np.exp(row - np.max(row))
                / np.exp(row - np.max(row)).sum()
                for row in logits
            ]
        )
        return float(
            -np.log(probabilities[np.arange(len(actions)), actions]).mean()
        )

    initial = graph.edge_gates.copy()
    graph.edge_gates[index] += epsilon
    plus = loss()
    graph.edge_gates[index] -= 2 * epsilon
    minus = loss()
    graph.edge_gates[...] = initial
    numerical = (plus - minus) / (2 * epsilon)
    graph.train_supervised(latents, actions)
    analytic = (initial[index] - graph.edge_gates[index]) / graph.learning_rate

    assert analytic == pytest.approx(numerical, rel=1e-4, abs=1e-7)


@pytest.mark.req("R-AGENT-GRAPH-03")
def test_closed_loop_policy_optimizer_applies_discounted_episode_rewards():
    """R-AGENT-GRAPH-03: update graph policy from a complete reward episode."""
    rng = np.random.default_rng(44)
    latents = rng.normal(size=(1, 3, 4))
    latents /= np.maximum(1.0, np.linalg.norm(latents, axis=-1, keepdims=True))
    graph = DifferentiableGatedGraph(
        node_count=3,
        latent_size=4,
        action_count=2,
        hidden_size=4,
        hops=1,
        seed=45,
    )
    policy = ClosedLoopRLPolicyOptimizer(graph, discount=0.9, seed=46)
    before = np.exp(graph.forward(latents).action_logits[0])
    before /= before.sum()

    policy.begin_episode()
    selected = policy.choose_action(latents)
    assert selected["action"] in {0, 1}
    assert selected["probabilities"].sum() == pytest.approx(1.0)
    assert policy.observe_reward(0.5, done=False) is None
    policy.choose_action(latents)
    result = policy.observe_reward(1.0, done=True)
    after = np.exp(graph.forward(latents).action_logits[0])
    after /= after.sum()

    assert result is not None
    assert result["updated_steps"] == 2
    assert result["episodes"] == 1
    assert result["algorithm"].startswith("discounted episodic REINFORCE")
    assert np.isfinite(after).all()
    assert not np.array_equal(before, after)


@pytest.mark.req("R-AGENT-GRAPH-03")
def test_positive_and_negative_policy_advantages_move_action_probability():
    """R-AGENT-GRAPH-03: move the selected action probability with reward sign."""
    rng = np.random.default_rng(50)
    latents = rng.normal(size=(1, 3, 4))
    latents /= np.maximum(1.0, np.linalg.norm(latents, axis=-1, keepdims=True))
    graph = DifferentiableGatedGraph(
        node_count=3,
        latent_size=4,
        action_count=2,
        hidden_size=4,
        hops=1,
        seed=51,
    )

    def action_zero_probability():
        return float(_probabilities(graph.forward(latents).action_logits[0])[0])

    initial = action_zero_probability()
    for _ in range(12):
        graph.policy_gradient_step(
            latents, 0, 1.0, learning_rate=0.01
        )
    rewarded = action_zero_probability()
    for _ in range(12):
        graph.policy_gradient_step(
            latents, 0, -1.0, learning_rate=0.01
        )
    penalized = action_zero_probability()

    assert rewarded > initial
    assert penalized < rewarded


@pytest.mark.req("R-AGENT-GRAPH-03")
def test_agent_graph_bridge_runs_user_models_then_returns_policy_actions(tmp_path):
    """R-AGENT-GRAPH-03: run individual trained agents through the latent graph."""
    agents = _trained_agent_network(tmp_path)
    bridge = AgentGraphBridge(agents, latent_size=8, action_count=3, seed=47)
    initial_updates = [
        node.model.reinforcement_updates for node in agents.nodes
    ]
    bridge.begin_episode()
    result = bridge.think("inspect this signal", max_new_tokens=8)

    assert len(result["messages"]) == 2
    assert {message["role"] for message in result["messages"]} == {
        "Extract relevant signals.",
        "Review the other node.",
    }
    assert len(result["action_logits"]) == 3
    action = bridge.choose_action()
    update = bridge.observe_reward(0.8, done=True)
    assert action["action"] in {0, 1, 2}
    assert update is not None
    assert update["updated_steps"] == 1
    assert [
        node.model.reinforcement_updates for node in agents.nodes
    ] == [count + 1 for count in initial_updates]


def _probabilities(logits):
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()
