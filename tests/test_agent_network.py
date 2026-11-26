import json

import pytest

from brna.agent_network import AgentNetwork
from brna.language_model import CharacterLanguageModel


@pytest.mark.req("R-AGENT-NET-01")
def test_positive_and_negative_episode_advantages_change_completion_likelihood():
    model = CharacterLanguageModel(hidden_size=8, seed=13)
    model.train("Question: hello\nAnswer: yes\n" * 4, epochs=3, sequence_length=32)
    prompt = "Question: hello\nAnswer:"
    completion = " yes"
    initial = model.completion_log_probability(prompt, completion)
    for _ in range(8):
        model.reinforce(prompt, completion, 1.0, learning_rate=0.01)
    increased = model.completion_log_probability(prompt, completion)
    assert increased > initial

    for _ in range(8):
        model.reinforce(prompt, completion, -1.0, learning_rate=0.01)
    decreased = model.completion_log_probability(prompt, completion)
    assert decreased < increased
    assert model.reinforcement_updates == 16


@pytest.mark.req("R-AGENT-NET-02")
def test_agents_share_messages_learn_from_episode_feedback_and_persist(tmp_path):
    network = AgentNetwork(tmp_path, seed=8)
    network.create_node("Planner", "Find a concise answer.", hidden_size=8, seed=1)
    network.create_node("Reviewer", "Check the answer for consistency.", hidden_size=8, seed=2)
    trained = network.train_all(
        "User: hello\nAssistant: a helpful local answer.\n" * 3,
        epochs=2,
        sequence_length=32,
    )
    assert trained["trained_nodes"] == 2
    result = network.think("hello", max_new_tokens=8)
    assert result["communication"] == "two-round all-to-all shared context"
    assert [message["round"] for message in result["messages"]] == [
        "private",
        "private",
        "shared",
        "shared",
    ]
    assert result["response"]

    update = network.reinforce(0.8, discount=0.9)
    assert update["updated_nodes"] == 2
    assert update["trajectory_steps"] == 4
    assert update["feedback_count"] == 1

    restored = AgentNetwork(tmp_path)
    assert [node["name"] for node in restored.summaries()] == ["Planner", "Reviewer"]
    assert restored.feedback_count == 1
    assert all(node["reinforcement_updates"] == 2 for node in restored.summaries())
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "run_records").glob("*.json")
    ]
    assert {record["run_type"] for record in records} == {
        "agent.create",
        "agent.train",
        "agent.collective_inference",
        "agent.episodic_reinforce",
    }
    training_record = next(record for record in records if record["run_type"] == "agent.train")
    assert training_record["settings"]["epochs"] == 2
    assert len(training_record["agent_seeds"]) == 2
    assert set(training_record["code_sha256"]) == {
        "agent_network.py",
        "language_model.py",
        "server.py",
        "reasoning_data.py",
    }


@pytest.mark.req("R-AGENT-NET-02")
def test_agent_network_rejects_collective_thinking_without_trained_nodes(tmp_path):
    network = AgentNetwork(tmp_path)
    network.create_node("Only node", "Answer questions.", hidden_size=8)
    with pytest.raises(ValueError, match="at least two"):
        network.think("hello")
