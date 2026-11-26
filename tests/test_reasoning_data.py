import json

import pytest

from brna.agent_network import AgentNetwork
from brna.reasoning_data import available_tasks, generate_examples, score_answer


@pytest.mark.req("R-REASONING-GYM-01")
def test_curated_reasoning_gym_tasks_generate_reproducible_verifiable_examples():
    tasks = available_tasks()
    assert set(tasks) == {"basic_arithmetic", "leg_counting"}

    first, first_dataset = generate_examples("basic_arithmetic", size=5, seed=41)
    second, _ = generate_examples("basic_arithmetic", size=5, seed=41)
    assert first == second
    assert len(first) == 5
    assert all(
        first_dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        for item in first
    )
    assert score_answer("basic_arithmetic", first[0]["answer"], seed=41) == 1.0
    assert score_answer("basic_arithmetic", "definitely wrong", seed=41) == 0.0

    legs, _ = generate_examples("leg_counting", size=2, seed=13)
    assert len(legs) == 2
    assert all(item["answer"].isdigit() for item in legs)

    with pytest.raises(ValueError, match="task_name"):
        generate_examples("unreviewed_task", size=1, seed=0)


@pytest.mark.req("R-REASONING-GYM-02")
def test_generated_dataset_trains_all_nodes_and_verifier_reward_updates_episode(tmp_path):
    network = AgentNetwork(tmp_path, seed=5)
    network.create_node("Arithmetic solver", "Solve arithmetic accurately.", hidden_size=8)
    network.create_node("Verifier", "Check peer arithmetic responses.", hidden_size=8)
    trained = network.train_reasoning_dataset(
        "basic_arithmetic",
        size=4,
        seed=7,
        epochs=2,
    )
    assert trained["dataset"] == "basic_arithmetic"
    assert trained["examples_generated"] == 4
    assert set(trained["node_losses"]) == {
        node["id"] for node in network.summaries()
    }

    episode = network.reasoning_episode(
        "basic_arithmetic",
        seed=9,
        index=0,
        max_new_tokens=8,
    )
    assert episode["task_name"] == "basic_arithmetic"
    assert 0.0 <= episode["verifier_score"] <= 1.0
    assert episode["reward"] == pytest.approx(2 * episode["verifier_score"] - 1)
    assert episode["learning"]["algorithm"] == (
        "episodic REINFORCE with discounted return baseline"
    )
    run_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "run_records").glob("*.json")
    ]
    dataset_run = next(
        record
        for record in run_records
        if record["run_type"] == "agent.train"
        and record["settings"]["source_type"] == "reasoning-gym:basic_arithmetic"
    )
    assert dataset_run["settings"]["source_details"] == {
        "task_name": "basic_arithmetic",
        "size": 4,
        "seed": 7,
    }
    reward_run = next(
        record
        for record in run_records
        if record["run_type"] == "agent.episodic_reinforce"
        and record["settings"]["reward_source"] == "reasoning-gym:basic_arithmetic"
    )
    assert reward_run["settings"]["reward_details"] == {
        "task_name": "basic_arithmetic",
        "seed": 9,
        "index": 0,
    }
