import base64
import json
import threading
import urllib.error
import urllib.request
import zipfile
from io import BytesIO

import pytest

from brna.server import StudioService, create_server


@pytest.fixture
def running_server(tmp_path):
    server = create_server(port=0, data_directory=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read())


@pytest.mark.req("R-STUDIO-02")
def test_server_serves_studio_and_reports_model_status(running_server):
    assert running_server.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(f"{running_server}/", timeout=3) as response:
        assert response.status == 200
        page = response.read()
        assert b"BRNA Studio" in page
        assert b"Agent network" in page

    status, health = _request(running_server, "/api/health")
    assert status == 200
    assert health["service"] == "brna-studio"
    assert health["model"]["trained"] is False


@pytest.mark.req("R-STUDIO-02")
def test_server_teaches_remembers_and_trains_local_model(running_server):
    status, result = _request(
        running_server,
        "/api/teach",
        {"prompt": "What is BRNA?", "response": "A message-passing network."},
    )
    assert status == 200
    assert result["conversation_examples"] == 1

    status, answer = _request(running_server, "/api/chat", {"prompt": "What is BRNA?"})
    assert status == 200
    assert answer == {
        "response": "A message-passing network.",
        "source": "taught-example",
        "similarity": 1.0,
    }

    status, training = _request(
        running_server,
        "/api/train",
        {"source": "examples", "epochs": 1, "sequence_length": 32},
    )
    assert status == 200
    assert training["trained"] is True
    assert training["training_characters"] > 0

    status, health = _request(running_server, "/api/health")
    assert status == 200
    assert health["model"]["trained"] is True


@pytest.mark.req("R-STUDIO-02")
def test_server_rejects_invalid_json_and_unknown_routes(running_server):
    request = urllib.request.Request(
        f"{running_server}/api/teach",
        data=b"{",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=3)
    assert error.value.code == 400
    assert "valid UTF-8 JSON" in error.value.read().decode("utf-8")

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{running_server}/api/missing", timeout=3)
    assert error.value.code == 404


@pytest.mark.req("R-UPLOAD-01")
def test_server_extracts_markdown_and_docx_for_training(running_server):
    markdown = b"# BRNA\nMessage passing learns from local text.\n"
    status, result = _request(
        running_server,
        "/api/upload",
        {
            "filename": "lesson.md",
            "content_base64": base64.b64encode(markdown).decode("ascii"),
            "epochs": 1,
            "sequence_length": 32,
        },
    )
    assert status == 200
    assert result["file_type"] == "text"
    assert result["extracted_characters"] == len(
        "# BRNA Message passing learns from local text."
    )

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Local document corpus</w:t></w:r></w:p></w:body>'
            "</w:document>",
        )
    status, result = _request(
        running_server,
        "/api/upload",
        {
            "filename": "lesson.docx",
            "content_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "epochs": 1,
            "sequence_length": 32,
        },
    )
    assert status == 200
    assert result["file_type"] == "docx"
    assert result["extracted_characters"] > 0


@pytest.mark.req("R-UPLOAD-01")
def test_server_rejects_unsupported_and_malformed_uploads(running_server):
    request = urllib.request.Request(
        f"{running_server}/api/upload",
        data=json.dumps(
            {
                "filename": "legacy.doc",
                "content_base64": base64.b64encode(b"legacy document").decode("ascii"),
                "epochs": 1,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=3)
    assert error.value.code == 400
    assert "not supported" in error.value.read().decode("utf-8")


@pytest.mark.req("R-AGENT-NET-03")
def test_server_builds_trains_and_reinforces_collective_agent_network(running_server):
    for name, role in (
        ("Planner", "Suggest a useful answer."),
        ("Reviewer", "Review the shared suggestions."),
    ):
        status, created = _request(
            running_server,
            "/api/agents/create",
            {"name": name, "role": role, "hidden_size": 8},
        )
        assert status == 200
        assert created["name"] == name

    status, training = _request(
        running_server,
        "/api/agents/train",
        {
            "text": "Question: hello\nAnswer: a locally learned response.\n" * 3,
            "epochs": 2,
            "sequence_length": 32,
        },
    )
    assert status == 200
    assert training["trained_nodes"] == 2
    assert set(training["node_losses"]) == {
        node["id"] for node in _request(running_server, "/api/agents")[1]["nodes"]
    }

    status, result = _request(
        running_server,
        "/api/agents/think",
        {"prompt": "hello", "max_new_tokens": 8},
    )
    assert status == 200
    assert len(result["messages"]) == 4
    assert result["communication"] == "two-round all-to-all shared context"

    status, feedback = _request(
        running_server,
        "/api/agents/feedback",
        {"reward": 0.7, "discount": 0.9},
    )
    assert status == 200
    assert feedback["updated_nodes"] == 2
    assert feedback["trajectory_steps"] == 4

    status, health = _request(running_server, "/api/health")
    assert status == 200
    assert health["agent_network"]["feedback_count"] == 1


@pytest.mark.req("R-AGENT-NET-04")
def test_server_trains_agent_nodes_from_uploaded_markdown(running_server):
    status, created = _request(
        running_server,
        "/api/agents/create",
        {"name": "Reader", "role": "Learn the supplied document.", "hidden_size": 8},
    )
    assert status == 200
    assert created["name"] == "Reader"
    content = b"# Local document\nAgents learn from this shared corpus."
    status, result = _request(
        running_server,
        "/api/agents/upload",
        {
            "filename": "shared.md",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "epochs": 1,
            "sequence_length": 32,
        },
    )
    assert status == 200
    assert result["trained_nodes"] == 1
    assert result["file_type"] == "text"
    assert result["extracted_characters"] > 0


@pytest.mark.req("R-REASONING-GYM-03")
def test_server_exposes_curated_reasoning_tasks_and_seeds_training(running_server):
    status, catalog = _request(running_server, "/api/agents/datasets")
    assert status == 200
    assert catalog["source"] == "open-thought/reasoning-gym"
    assert set(catalog["tasks"]) == {"basic_arithmetic", "leg_counting"}

    for name in ("Arithmetic", "Reviewer"):
        status, _ = _request(
            running_server,
            "/api/agents/create",
            {"name": name, "role": "Solve arithmetic tasks.", "hidden_size": 8},
        )
        assert status == 200

    payload = {
        "task_name": "basic_arithmetic",
        "size": 3,
        "seed": 12,
        "epochs": 1,
    }
    status, first = _request(running_server, "/api/agents/datasets/train", payload)
    assert status == 200
    assert first["examples_generated"] == 3
    assert first["seed"] == 12

    second_payload = {**payload}
    status, second = _request(running_server, "/api/agents/datasets/train", second_payload)
    assert status == 200
    assert second["dataset"] == first["dataset"]
    assert second["examples_generated"] == first["examples_generated"]
    assert second["seed"] == first["seed"]

    status, episode = _request(
        running_server,
        "/api/agents/reasoning-episode",
        {
            "task_name": "basic_arithmetic",
            "seed": 22,
            "index": 0,
            "max_new_tokens": 8,
        },
    )
    assert status == 200
    assert 0 <= episode["verifier_score"] <= 1
    assert episode["reward"] == pytest.approx(2 * episode["verifier_score"] - 1)


@pytest.mark.req("R-AGENT-GRAPH-04")
def test_server_trains_and_runs_user_agents_through_persisted_latent_graph(
    running_server, tmp_path
):
    for name, role in (
        ("Sensor", "Extract relevant signals."),
        ("Critic", "Review the other node."),
    ):
        status, _ = _request(
            running_server,
            "/api/agents/create",
            {"name": name, "role": role, "hidden_size": 8},
        )
        assert status == 200
    status, _ = _request(
        running_server,
        "/api/agents/train",
        {
            "text": "User: signal\nAssistant: detect a stable pattern.\n" * 4,
            "epochs": 2,
            "sequence_length": 32,
        },
    )
    assert status == 200

    status, report = _request(
        running_server,
        "/api/agents/graph/train",
        {
            "examples": [
                {"contexts": ["detect a signal", "review a signal"], "action": 0},
                {
                    "contexts": ["detect another signal", "review another signal"],
                    "action": 1,
                },
            ],
            "epochs": 2,
        },
    )
    assert status == 200
    assert report["examples_trained"] == 2
    assert report["updates"] == 2

    status, graph_status = _request(running_server, "/api/agents/graph")
    assert status == 200
    assert graph_status["latent_size"] == 8
    assert graph_status["active_edges"] > 0

    status, started = _request(
        running_server, "/api/agents/graph/episode/start", {}
    )
    assert status == 200
    assert started["episode_started"] is True
    status, step = _request(
        running_server,
        "/api/agents/graph/step",
        {"prompt": "evaluate this signal", "max_new_tokens": 8},
    )
    assert status == 200
    assert len(step["messages"]) == 2
    assert step["action"]["action"] in range(4)
    status, pending = _request(running_server, "/api/agents/graph")
    assert status == 200
    assert pending["awaiting_reward"] is True
    status, reward = _request(
        running_server,
        "/api/agents/graph/reward",
        {"reward": 0.8, "done": True},
    )
    assert status == 200
    assert reward["episode_complete"] is True
    assert reward["updated_steps"] == 1
    assert list((tmp_path / "agent-network" / "graphs").glob("*.npz"))
    restored = StudioService(tmp_path).agent_graph_status()
    assert restored["episodes_trained"] == 1
    assert restored["reward_baseline"] == pytest.approx(0.8)


@pytest.mark.req("R-HF-01")
def test_local_backbone_training_and_chat_use_the_bounded_agent_graph(
    tmp_path, monkeypatch
):
    """R-HF-01: use a local pretrained-backbone adapter through graph chat."""
    import numpy as np

    import brna.server as server_module

    class FakeBackbone:
        class FakePrefixGraph:
            def __init__(self):
                self.last_contexts = None

            def __call__(self, contexts):
                self.last_contexts = contexts
                states = np.tanh(np.asarray(contexts, dtype=float))
                norms = np.linalg.norm(states, axis=-1, keepdims=True)
                states /= np.maximum(norms, 1.0)
                return {
                    "latents": states,
                    "prefix_embeddings": states[:, None, :],
                }

            def parameters(self):
                return []

            def state_dict(self):
                return {}

        def __init__(self, model_directory, adapter_directory, *, device):
            self.hidden_size = 8
            self.device = "cpu" if device == "auto" else device
            self.adapter_directory = adapter_directory
            self.trained = set()
            self.prefixes_seen = []

        def create_soft_prefix_graph(
            self, *, node_count, adjacency, latent_size, hops, seed
        ):
            return self.FakePrefixGraph()

        def ensure_adapter(self, adapter_id, role):
            return f"node_{adapter_id}"

        def train_text(
            self, adapter_id, role, text, *, epochs=1, learning_rate=2e-4
        ):
            self.trained.add(adapter_id)
            return [0.4] * epochs

        def generate(
            self,
            prompt,
            *,
            adapter_id=None,
            role=None,
            max_new_tokens=128,
            temperature=0.7,
            soft_prefix=None,
        ):
            self.prefixes_seen.append(soft_prefix)
            return f"Generated by {role.rstrip('.')}."

        def encode_context(self, text, *, adapter_id, role):
            return np.full(self.hidden_size, len(text) / 100.0)

        def encode_context_tensor(self, text, *, adapter_id, role):
            return self.encode_context(text, adapter_id=adapter_id, role=role)

        def stack_contexts(self, contexts):
            return np.stack(contexts)

        def tensor_to_numpy(self, values):
            return np.asarray(values)

        def grpo_group_update(
            self,
            adapter_id,
            role,
            prompt,
            completions,
            rewards,
            *,
            soft_prefix,
            graph_parameters,
            learning_rate,
            kl_beta,
        ):
            self.prefixes_seen.append(soft_prefix)
            self.trained.add(adapter_id)
            return {
                "loss": 0.1,
                "mean_kl": 0.02,
                "mean_reward": sum(rewards) / len(rewards),
                "reward_std": float(np.std(rewards)),
                "group_size": len(completions),
            }

        def reinforce(
            self,
            adapter_id,
            role,
            prompt,
            completion,
            advantage,
            *,
            learning_rate,
        ):
            return -advantage

        def has_trained_adapter(self, adapter_id):
            return adapter_id in self.trained

    monkeypatch.setattr(server_module, "LocalLoRABackbone", FakeBackbone)
    service = StudioService(tmp_path, backbone_directory=tmp_path)
    service.create_agent(
        {"name": "Researcher", "role": "Analyze evidence.", "hidden_size": 8}
    )
    training = service.train_agents(
        {"text": "Useful local training material.", "epochs": 2}
    )
    assert training["trained_nodes"] == 1
    assert service.health()["model"]["backend"] == "local-pretrained-causal-lm"

    result = service.chat("Summarize this research.")
    assert result["source"] == "local-pretrained-agent-graph"
    assert result["response"] == "Generated by Analyze evidence."
    assert result["messages"][0]["latent"]
    assert service.backbone.prefixes_seen
    assert service.backbone.prefixes_seen[-1] is not None


@pytest.mark.req("R-HF-03")
def test_local_backbone_runs_verifier_scored_group_relative_episode(
    tmp_path, monkeypatch
):
    """R-HF-03: optimize verifier-scored groups and persist their prefix graph."""
    import numpy as np

    import brna.agent_graph as graph_module
    import brna.server as server_module

    class FakeBackbone:
        class FakePrefixGraph:
            def __call__(self, contexts):
                states = np.tanh(np.asarray(contexts, dtype=float))
                norms = np.linalg.norm(states, axis=-1, keepdims=True)
                states /= np.maximum(norms, 1.0)
                return {
                    "latents": states,
                    "prefix_embeddings": states[:, None, :],
                }

            def parameters(self):
                return []

            def state_dict(self):
                return {}

        def __init__(self, model_directory, adapter_directory, *, device):
            self.hidden_size = 8
            self.device = "cpu"
            self.adapter_directory = adapter_directory

        def ensure_adapter(self, adapter_id, role):
            return f"node_{adapter_id}"

        def train_text(
            self, adapter_id, role, text, *, epochs=1, learning_rate=2e-4
        ):
            return [0.4] * epochs

        def create_soft_prefix_graph(
            self, *, node_count, adjacency, latent_size, hops, seed
        ):
            return self.FakePrefixGraph()

        def encode_context_tensor(self, text, *, adapter_id, role):
            return np.full(self.hidden_size, len(text) / 100.0)

        def stack_contexts(self, contexts):
            return np.stack(contexts)

        def tensor_to_numpy(self, values):
            return np.asarray(values)

        def generate(
            self,
            prompt,
            *,
            adapter_id=None,
            role=None,
            max_new_tokens=128,
            temperature=0.7,
            soft_prefix=None,
        ):
            assert soft_prefix is not None
            return "42"

        def grpo_group_update(
            self,
            adapter_id,
            role,
            prompt,
            completions,
            rewards,
            *,
            soft_prefix,
            graph_parameters,
            learning_rate,
            kl_beta,
        ):
            assert soft_prefix is not None
            return {
                "loss": 0.0,
                "mean_kl": 0.0,
                "mean_reward": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "group_size": len(completions),
            }

    class FakeDataset:
        def score_answer(self, *, answer, entry):
            return 1.0 if answer == entry["answer"] else 0.0

    monkeypatch.setattr(server_module, "LocalLoRABackbone", FakeBackbone)
    monkeypatch.setattr(
        graph_module,
        "generate_examples",
        lambda task_name, *, size, seed: (
            [{"question": "What is 6 times 7?", "answer": "42"}],
            FakeDataset(),
        ),
    )
    service = StudioService(tmp_path, backbone_directory=tmp_path)
    for name in ("Arithmetic", "Reviewer"):
        service.create_agent(
            {"name": name, "role": "Solve arithmetic.", "hidden_size": 8}
        )
    service.train_agents({"text": "Arithmetic training text.", "epochs": 1})

    result = service.run_agent_grpo_episode(
        {"task_name": "basic_arithmetic", "seed": 7, "index": 0, "group_size": 2}
    )
    assert result["verified_candidates"] == 4
    assert result["mean_reward"] == 1.0
    assert result["algorithm"].startswith("group-relative")
    prefix_checkpoints = list(
        (tmp_path / "agent-network" / "graphs").glob("*.prefix.npz")
    )
    assert prefix_checkpoints
    prefix_checkpoints[0].unlink()
    service._agent_graph_bridge = None
    with pytest.raises(ValueError, match="missing its soft-prefix parameters"):
        service.agent_graph_status()
