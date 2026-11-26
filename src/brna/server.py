"""Local web/API host for BRNA Studio and its self-trained text model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from brna.agent_graph import AgentGraphBridge
from brna.agent_network import AgentNetwork
from brna.conversation import ConversationMemory
from brna.data_ingestion import extract_training_text
from brna.language_model import CharacterLanguageModel
from brna.reasoning_data import available_tasks

LOGGER = logging.getLogger("brna.server")
MAX_REQUEST_BYTES = 27_000_000
MAX_PROMPT_CHARACTERS = 4_000
MAX_RESPONSE_TOKENS = 512


class StudioService:
    """Owns the local conversation memory and trainable language model."""

    def __init__(self, data_directory: Path, web_directory: Path | None = None) -> None:
        self.data_directory = data_directory.expanduser().resolve()
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.web_directory = (web_directory or _web_directory()).resolve()
        if not (self.web_directory / "index.html").is_file():
            raise FileNotFoundError(f"BRNA Studio browser assets were not found in {self.web_directory}")
        self.memory_path = self.data_directory / "conversation-memory.json"
        self.model_path = self.data_directory / "language-model.npz"
        self.lock = threading.RLock()
        self.memory = self._load_memory()
        self.agent_network = AgentNetwork(self.data_directory / "agent-network")
        self._agent_graph_bridge: AgentGraphBridge | None = None
        self._agent_graph_node_ids: tuple[str, ...] = ()
        self._agent_graph_checkpoint: Path | None = None
        self.model = (
            CharacterLanguageModel.load(self.model_path)
            if self.model_path.exists()
            else CharacterLanguageModel(hidden_size=32, seed=7)
        )

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "service": "brna-studio",
                "ready": True,
                "model": {
                    "trained": self.model.trained_characters > 0,
                    "training_characters": self.model.trained_characters,
                    "hidden_size": self.model.hidden_size,
                    "vocabulary": "Unicode characters",
                },
                "conversation_examples": len(self.memory.examples),
                "agent_network": {
                    "nodes": self.agent_network.summaries(),
                    "shared_training_characters": len(self.agent_network.shared_corpus),
                    "feedback_count": self.agent_network.feedback_count,
                    "reward_baseline": self.agent_network.reward_baseline,
                },
            }

    def teach(self, prompt: str, response: str) -> dict[str, Any]:
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError("prompt and response must be strings")
        if not prompt.strip() or not response.strip():
            raise ValueError("both a prompt and response are required")
        if len(prompt) > MAX_PROMPT_CHARACTERS or len(response) > MAX_PROMPT_CHARACTERS * 2:
            raise ValueError("prompt or response is too long")
        with self.lock:
            duplicate = any(
                example.prompt == prompt.strip() and example.response == response.strip()
                for example in self.memory.examples
            )
            self.memory.teach(prompt, response)
            network_result = None
            if not duplicate:
                network_result = self.agent_network.teach(prompt, response)
            self._save_memory()
            result = {"saved": True, "conversation_examples": len(self.memory.examples)}
            if network_result is not None:
                result["agent_network"] = network_result
            return result

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self.agent_network.create_node(
                payload.get("name"),
                payload.get("role"),
                hidden_size=payload.get("hidden_size", 32),
                seed=payload.get("seed"),
            )

    def train_agents(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "text" in payload:
            text = payload["text"]
        elif payload.get("source") == "examples":
            with self.lock:
                text = "\n".join(
                    f"User: {example.prompt}\nAssistant: {example.response}"
                    for example in self.memory.examples
                )
        else:
            raise ValueError("provide training text or set source to 'examples'")
        if not isinstance(text, str):
            raise TypeError("training text must be a string")
        with self.lock:
            return self.agent_network.train_all(
                text,
                epochs=payload.get("epochs", 3),
                learning_rate=payload.get("learning_rate", 0.05),
                sequence_length=payload.get("sequence_length", 96),
                source_type=payload.get("source_type", "provided-text"),
            )

    def reasoning_tasks(self) -> dict[str, Any]:
        return {"source": "open-thought/reasoning-gym", "tasks": available_tasks()}

    def train_reasoning_agents(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self.agent_network.train_reasoning_dataset(
                payload.get("task_name"),
                size=payload.get("size", 16),
                seed=payload.get("seed", 0),
                epochs=payload.get("epochs", 3),
            )

    def run_reasoning_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self.agent_network.reasoning_episode(
                payload.get("task_name"),
                seed=payload.get("seed", 0),
                index=payload.get("index", 0),
                max_new_tokens=payload.get("max_new_tokens", 48),
                discount=payload.get("discount", 0.95),
                learning_rate=payload.get("learning_rate", 0.01),
            )

    def upload_agent_training_file(self, payload: dict[str, Any]) -> dict[str, Any]:
        filename = payload.get("filename")
        encoded_content = payload.get("content_base64")
        text, kind = extract_training_text(filename, encoded_content)
        with self.lock:
            result = self.agent_network.train_all(
                text,
                epochs=payload.get("epochs", 3),
                learning_rate=payload.get("learning_rate", 0.05),
                sequence_length=payload.get("sequence_length", 96),
                source_type=f"upload:{kind}",
            )
            result.update(
                {
                    "filename": filename,
                    "file_type": kind,
                    "extracted_characters": len(text),
                }
            )
            return result

    def think_with_agents(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self.agent_network.think(
                payload.get("prompt"),
                max_new_tokens=payload.get("max_new_tokens", 48),
            )

    def reinforce_agents(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self.agent_network.reinforce(
                payload.get("reward"),
                discount=payload.get("discount", 0.95),
                learning_rate=payload.get("learning_rate", 0.01),
            )

    def agent_graph_status(self) -> dict[str, Any]:
        with self.lock:
            bridge = self._get_agent_graph_bridge()
            return {
                "node_ids": [node.agent.agent_id for node in bridge.nodes],
                "roles": [node.agent.role for node in bridge.nodes],
                "latent_size": bridge.graph.latent_size,
                "action_count": bridge.graph.action_count,
                "hops": bridge.graph.hops,
                "graph_parameters": bridge.graph.parameter_count,
                "active_edges": int(bridge.graph.adjacency.sum()),
                "episode_active": bridge.policy.episode_active,
                "awaiting_reward": bridge.policy.awaiting_reward,
                "episodes_trained": bridge.policy.episode_count,
                "reward_baseline": bridge.policy.baseline,
            }

    def start_agent_graph_episode(self) -> dict[str, Any]:
        with self.lock:
            bridge = self._get_agent_graph_bridge()
            bridge.begin_episode()
            return {"episode_started": True}

    def step_agent_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            bridge = self._get_agent_graph_bridge()
            result = bridge.think(
                payload.get("prompt"),
                max_new_tokens=payload.get("max_new_tokens", 32),
            )
            action = bridge.choose_action()
            action["probabilities"] = action["probabilities"].tolist()
            result["action"] = action
            return result

    def reward_agent_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            bridge = self._get_agent_graph_bridge()
            result = bridge.observe_reward(
                payload.get("reward"),
                done=payload.get("done", False),
            )
            if result is None:
                return {"episode_complete": False, "reward_recorded": True}
            if self._agent_graph_checkpoint is None:
                raise RuntimeError("agent graph checkpoint path is unavailable")
            bridge.save_checkpoint(self._agent_graph_checkpoint)
            return {"episode_complete": True, **result}

    def train_agent_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        epochs = payload.get("epochs", 1)
        if not isinstance(examples, list) or not 1 <= len(examples) <= 256:
            raise ValueError("examples must be a list containing from 1 to 256 items")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
            raise ValueError("epochs must be from 1 to 100")
        with self.lock:
            bridge = self._get_agent_graph_bridge()
            if bridge.policy.episode_active:
                raise RuntimeError("finish the active graph-policy episode before training")
            latent_rows = []
            actions = []
            for index, example in enumerate(examples):
                if not isinstance(example, dict):
                    raise TypeError(f"examples[{index}] must be an object")
                contexts = example.get("contexts")
                if (
                    not isinstance(contexts, list)
                    or len(contexts) != len(bridge.nodes)
                    or any(not isinstance(context, str) for context in contexts)
                ):
                    raise ValueError(
                        f"examples[{index}].contexts must provide one text context per agent"
                    )
                latent_rows.append(
                    np.stack(
                        [
                            node.encode_text(context)
                            for node, context in zip(
                                bridge.nodes, contexts, strict=True
                            )
                        ]
                    )
                )
                actions.append(example.get("action"))
            losses = bridge.trainer.fit(
                np.stack(latent_rows),
                np.asarray(actions),
                epochs=epochs,
            )
            if self._agent_graph_checkpoint is None:
                raise RuntimeError("agent graph checkpoint path is unavailable")
            bridge.save_checkpoint(self._agent_graph_checkpoint)
            return {
                "examples_trained": len(examples),
                "action_count": bridge.graph.action_count,
                **losses,
            }

    def _get_agent_graph_bridge(self) -> AgentGraphBridge:
        nodes = tuple(node.agent_id for node in self.agent_network.nodes)
        if not nodes:
            raise ValueError("create at least one model node before using the latent graph")
        if (
            self._agent_graph_bridge is not None
            and self._agent_graph_node_ids == nodes
        ):
            return self._agent_graph_bridge

        bridge = AgentGraphBridge(self.agent_network, seed=17)
        key = hashlib.sha256("\n".join(nodes).encode("utf-8")).hexdigest()[:16]
        checkpoint = (
            self.agent_network.directory / "graphs" / f"{key}-latent-graph.npz"
        )
        if checkpoint.exists():
            bridge.load_checkpoint(checkpoint)
        self._agent_graph_bridge = bridge
        self._agent_graph_node_ids = nodes
        self._agent_graph_checkpoint = checkpoint
        return bridge

    def train(self, payload: dict[str, Any]) -> dict[str, Any]:
        epochs = payload.get("epochs", 3)
        learning_rate = payload.get("learning_rate", 0.05)
        sequence_length = payload.get("sequence_length", 96)
        if "text" in payload:
            text = payload["text"]
        elif payload.get("source") == "examples":
            with self.lock:
                text = "\n".join(
                    f"User: {example.prompt}\nAssistant: {example.response}"
                    for example in self.memory.examples
                )
        else:
            raise ValueError("provide training text or set source to 'examples'")
        if not isinstance(text, str):
            raise TypeError("training text must be a string")
        with self.lock:
            losses = self.model.train(
                text,
                epochs=epochs,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
            )
            self.model.save(self.model_path)
            return {
                "trained": True,
                "training_characters": self.model.trained_characters,
                "epochs": len(losses),
                "losses": losses,
            }

    def chat(self, prompt: str, *, max_new_tokens: int = 240) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(f"prompt must not exceed {MAX_PROMPT_CHARACTERS} characters")
        with self.lock:
            match = self.memory.retrieve(prompt, min_similarity=0.75)
            if match:
                response, similarity = match
                return {"response": response, "source": "taught-example", "similarity": similarity}
            response = self.model.generate(
                f"User: {prompt}\nAssistant:",
                max_new_tokens=max_new_tokens,
            )
            if not response:
                raise RuntimeError("The model generated an empty response; train it with more text and try again.")
            return {"response": response, "source": "local-model"}

    def upload_training_file(self, payload: dict[str, Any]) -> dict[str, Any]:
        filename = payload.get("filename")
        encoded_content = payload.get("content_base64")
        text, kind = extract_training_text(filename, encoded_content)
        epochs = payload.get("epochs", 3)
        learning_rate = payload.get("learning_rate", 0.05)
        sequence_length = payload.get("sequence_length", 96)
        with self.lock:
            losses = self.model.train(
                text,
                epochs=epochs,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
            )
            self.model.save(self.model_path)
            return {
                "trained": True,
                "filename": filename,
                "file_type": kind,
                "training_characters": self.model.trained_characters,
                "extracted_characters": len(text),
                "epochs": len(losses),
                "losses": losses,
            }

    def _load_memory(self) -> ConversationMemory:
        if not self.memory_path.exists():
            return ConversationMemory()
        try:
            return ConversationMemory.from_dict(json.loads(self.memory_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"could not load saved conversation memory: {exc}") from exc

    def _save_memory(self) -> None:
        encoded = json.dumps(self.memory.to_dict(), ensure_ascii=False, allow_nan=False)
        _atomic_write(self.memory_path, encoded.encode("utf-8"))


class StudioRequestHandler(SimpleHTTPRequestHandler):
    """Serve the browser app and bounded JSON endpoints from one local origin."""

    service: StudioService

    def __init__(self, *args: Any, service: StudioService, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, directory=str(service.web_directory), **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json(200, self.service.health())
        elif path == "/api/examples":
            with self.service.lock:
                self._send_json(200, self.service.memory.to_dict())
        elif path == "/api/agents":
            with self.service.lock:
                self._send_json(
                    200,
                    {
                        "nodes": self.service.agent_network.summaries(),
                        "shared_training_characters": len(
                            self.service.agent_network.shared_corpus
                        ),
                        "feedback_count": self.service.agent_network.feedback_count,
                        "reward_baseline": self.service.agent_network.reward_baseline,
                    },
                )
        elif path == "/api/agents/graph":
            try:
                self._send_json(200, self.service.agent_graph_status())
            except (RuntimeError, ValueError) as exc:
                self._send_json(409, {"error": str(exc)})
        elif path == "/api/agents/datasets":
            try:
                self._send_json(200, self.service.reasoning_tasks())
            except RuntimeError as exc:
                self._send_json(503, {"error": str(exc)})
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "API endpoint not found"})
        else:
            super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        routes = {
            "/api/chat",
            "/api/teach",
            "/api/train",
            "/api/upload",
            "/api/agents/create",
            "/api/agents/train",
            "/api/agents/upload",
            "/api/agents/datasets/train",
            "/api/agents/reasoning-episode",
            "/api/agents/think",
            "/api/agents/feedback",
            "/api/agents/graph/episode/start",
            "/api/agents/graph/step",
            "/api/agents/graph/reward",
            "/api/agents/graph/train",
        }
        if path not in routes:
            self._send_json(404, {"error": "API endpoint not found"})
            return
        try:
            payload = self._read_json()
            if path == "/api/chat":
                result = self.service.chat(
                    payload.get("prompt"),
                    max_new_tokens=payload.get("max_new_tokens", 240),
                )
            elif path == "/api/teach":
                result = self.service.teach(payload.get("prompt"), payload.get("response"))
            elif path == "/api/agents/create":
                result = self.service.create_agent(payload)
            elif path == "/api/agents/train":
                result = self.service.train_agents(payload)
            elif path == "/api/agents/upload":
                result = self.service.upload_agent_training_file(payload)
            elif path == "/api/agents/datasets/train":
                result = self.service.train_reasoning_agents(payload)
            elif path == "/api/agents/reasoning-episode":
                result = self.service.run_reasoning_episode(payload)
            elif path == "/api/agents/think":
                result = self.service.think_with_agents(payload)
            elif path == "/api/agents/feedback":
                result = self.service.reinforce_agents(payload)
            elif path == "/api/agents/graph/episode/start":
                result = self.service.start_agent_graph_episode()
            elif path == "/api/agents/graph/step":
                result = self.service.step_agent_graph(payload)
            elif path == "/api/agents/graph/reward":
                result = self.service.reward_agent_graph(payload)
            elif path == "/api/agents/graph/train":
                result = self.service.train_agent_graph(payload)
            else:
                result = (
                    self.service.upload_training_file(payload)
                    if path == "/api/upload"
                    else self.service.train(payload)
                )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"request body must be valid UTF-8 JSON: {exc}"})
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
        else:
            self._send_json(200, result)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be a whole number") from exc
        if not 1 <= length <= MAX_REQUEST_BYTES:
            raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise TypeError("request body must be a JSON object")
        return data

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    data_directory: Path | None = None,
    web_directory: Path | None = None,
) -> ThreadingHTTPServer:
    service = StudioService(data_directory or _default_data_directory(), web_directory)

    class BoundHandler(StudioRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, service=service, **kwargs)

    return ThreadingHTTPServer((host, port), BoundHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-hosted BRNA Studio and local language model.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (defaults to localhost only)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (defaults to 8765)")
    parser.add_argument("--data-dir", type=Path, help="model and conversation storage directory")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_directory = (args.data_dir or _default_data_directory()).expanduser().resolve()
    server = create_server(host=args.host, port=args.port, data_directory=data_directory)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"BRNA Studio is running at {url}")
    print(f"Local model data is stored in {data_directory}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BRNA Studio.")
    finally:
        server.server_close()


def _web_directory() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    candidates = []
    if bundle_root:
        candidates.append(Path(bundle_root) / "web")
    candidates.extend(
        (
            Path(__file__).resolve().parents[2] / "web",
            Path(sys.prefix) / "share" / "brna-studio" / "web",
        )
    )
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    raise FileNotFoundError("BRNA Studio web assets were not found in the app bundle or Python installation")


def _default_data_directory() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "BRNA Studio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "BRNA Studio"
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "brna-studio"


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
