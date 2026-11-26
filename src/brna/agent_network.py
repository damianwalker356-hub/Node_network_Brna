"""Cooperative role-specialized language-model agents with episodic feedback."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brna.language_model import (
    MAX_SEQUENCE_LENGTH,
    MAX_TRAINING_CHARACTERS,
    MAX_VOCABULARY_SIZE,
    CharacterLanguageModel,
)
from brna.reasoning_data import TASKS, generate_examples, score_answer

MAX_AGENT_NODES = 8
MAX_AGENT_NAME_CHARACTERS = 48
MAX_AGENT_ROLE_CHARACTERS = 240
MAX_COLLECTIVE_PROMPT_CHARACTERS = 4_000
MAX_AGENT_RESPONSE_TOKENS = 128


@dataclass
class AgentNode:
    agent_id: str
    name: str
    role: str
    seed: int
    model: CharacterLanguageModel

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "seed": self.seed,
            "hidden_size": self.model.hidden_size,
            "trained_characters": self.model.trained_characters,
            "reinforcement_updates": self.model.reinforcement_updates,
        }


class AgentNetwork:
    """A bounded all-to-all team whose members are independently trained models.

    Each turn has a private first pass followed by a shared pass. Every node in
    the shared pass sees all private-pass messages and every earlier shared-pass
    message. The final node's shared-pass answer is the team's final response.
    """

    def __init__(self, directory: str | Path, *, seed: int = 17) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_records_directory = self.directory / "run_records"
        self.metadata_path = self.directory / "network.json"
        self.shared_corpus = ""
        self.reward_baseline = 0.0
        self.feedback_count = 0
        self.nodes: list[AgentNode] = []
        self._last_episode: list[dict[str, str]] | None = None
        self._seed = seed
        if self.metadata_path.exists():
            self._load()

    def create_node(
        self,
        name: str,
        role: str,
        *,
        hidden_size: int = 32,
        seed: int | None = None,
    ) -> dict[str, Any]:
        name = _validated_text(name, "name", MAX_AGENT_NAME_CHARACTERS)
        role = _validated_text(role, "role", MAX_AGENT_ROLE_CHARACTERS)
        if len(self.nodes) >= MAX_AGENT_NODES:
            raise ValueError(f"agent network cannot contain more than {MAX_AGENT_NODES} nodes")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative whole number")
        agent_id = uuid.uuid4().hex
        node_seed = self._seed + len(self.nodes) if seed is None else seed
        model = CharacterLanguageModel(
            hidden_size=hidden_size,
            seed=node_seed,
        )
        if self.shared_corpus:
            model.train(self.shared_corpus, epochs=1, sequence_length=96)
        node = AgentNode(agent_id, name, role, node_seed, model)
        self.nodes.append(node)
        self._save_node(node)
        self._save_metadata()
        self._last_episode = None
        self._record_run(
            "agent.create",
            {"hidden_size": hidden_size, "role_characters": len(role)},
            {"agent_id": agent_id, "name": name},
            nodes=[node],
        )
        return node.summary()

    def summaries(self) -> list[dict[str, Any]]:
        return [node.summary() for node in self.nodes]

    def bounded_latent_nodes(
        self,
        *,
        latent_size: int = 8,
        max_norm: float = 1.0,
    ):
        """Wrap each trained text agent for bounded latent graph communication."""
        from brna.agent_graph import BoundedLatentNode

        return [
            BoundedLatentNode(node, latent_size=latent_size, max_norm=max_norm)
            for node in self.nodes
        ]

    def save_models(self) -> None:
        for node in self.nodes:
            self._save_node(node)

    def train_all(
        self,
        text: str,
        *,
        epochs: int = 3,
        learning_rate: float = 0.05,
        sequence_length: int = 96,
        source_type: str = "provided-text",
        source_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.nodes:
            raise ValueError("create at least one agent node before training")
        if not isinstance(text, str):
            raise TypeError("training text must be a string")
        if not text.strip():
            raise ValueError("training text must not be empty")
        if len(text) > MAX_TRAINING_CHARACTERS:
            raise ValueError(
                f"training text must not exceed {MAX_TRAINING_CHARACTERS} characters"
            )
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be a whole number from 1 to 50")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1
        ):
            raise ValueError("learning_rate must be finite and between 0 and 1")
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or not 8 <= sequence_length <= MAX_SEQUENCE_LENGTH
        ):
            raise ValueError(f"sequence_length must be from 8 to {MAX_SEQUENCE_LENGTH}")
        known_source = source_type in {
            "provided-text",
            "conversation",
            "upload:text",
            "upload:docx",
            "upload:pdf",
        }
        known_reasoning_task = (
            isinstance(source_type, str)
            and source_type.startswith("reasoning-gym:")
            and source_type.removeprefix("reasoning-gym:") in TASKS
        )
        if not isinstance(source_type, str) or not (known_source or known_reasoning_task):
            raise ValueError("unsupported training source type")
        if source_details is not None and not isinstance(source_details, dict):
            raise TypeError("source_details must be a mapping")
        if any(
            len(set(node.model.vocabulary).union(text)) > MAX_VOCABULARY_SIZE
            for node in self.nodes
        ):
            raise ValueError(
                f"model vocabulary cannot exceed {MAX_VOCABULARY_SIZE} characters"
            )
        candidate_corpus = f"{self.shared_corpus}\n{text}".strip()
        if len(candidate_corpus) > MAX_TRAINING_CHARACTERS:
            raise ValueError(
                f"shared training corpus cannot exceed {MAX_TRAINING_CHARACTERS} characters"
            )
        losses: dict[str, list[float]] = {}
        for node in self.nodes:
            losses[node.agent_id] = node.model.train(
                text,
                epochs=epochs,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
            )
            self._save_node(node)
        self.shared_corpus = candidate_corpus
        self._save_metadata()
        self._last_episode = None
        result = {
            "trained_nodes": len(self.nodes),
            "training_characters": len(self.shared_corpus),
            "node_losses": losses,
        }
        self._record_run(
            "agent.train",
            {
                "source_type": source_type,
                "source_details": source_details or {},
                "epochs": epochs,
                "learning_rate": learning_rate,
                "sequence_length": sequence_length,
                "input_characters": len(text),
            },
            {
                "trained_nodes": len(self.nodes),
                "shared_training_characters": len(self.shared_corpus),
                "node_losses": losses,
            },
        )
        return result

    def train_reasoning_dataset(
        self,
        task_name: str,
        *,
        size: int = 16,
        seed: int = 0,
        epochs: int = 3,
    ) -> dict[str, Any]:
        examples, _ = generate_examples(task_name, size=size, seed=seed)
        text = "\n".join(
            f"User: {example['question']}\nAssistant: {example['answer']}"
            for example in examples
        )
        result = self.train_all(
            text,
            epochs=epochs,
            source_type=f"reasoning-gym:{task_name}",
            source_details={"task_name": task_name, "size": size, "seed": seed},
        )
        result["dataset"] = task_name
        result["examples_generated"] = len(examples)
        result["seed"] = seed
        return result

    def reasoning_episode(
        self,
        task_name: str,
        *,
        seed: int = 0,
        index: int = 0,
        max_new_tokens: int = 48,
        discount: float = 0.95,
        learning_rate: float = 0.01,
    ) -> dict[str, Any]:
        examples, _ = generate_examples(task_name, size=index + 1, seed=seed)
        entry = examples[index]
        run_details = {"task_name": task_name, "seed": seed, "index": index}
        result = self.think(
            entry["question"],
            max_new_tokens=max_new_tokens,
            run_details=run_details,
        )
        verifier_score = score_answer(
            task_name,
            result["response"],
            seed=seed,
            index=index,
        )
        reward = 2.0 * verifier_score - 1.0
        update = self.reinforce(
            reward,
            discount=discount,
            learning_rate=learning_rate,
            reward_source=f"reasoning-gym:{task_name}",
            verifier_score=verifier_score,
            reward_details=run_details,
        )
        return {
            "task_name": task_name,
            "seed": seed,
            "index": index,
            "question": entry["question"],
            "expected_answer": entry["answer"],
            "model_response": result["response"],
            "verifier_score": verifier_score,
            "reward_details": run_details,
            "reward": reward,
            "learning": update,
        }

    def teach(
        self,
        prompt: str,
        response: str,
        *,
        epochs: int = 1,
        learning_rate: float = 0.05,
    ) -> dict[str, Any]:
        prompt = _validated_text(prompt, "prompt", MAX_COLLECTIVE_PROMPT_CHARACTERS)
        response = _validated_text(response, "response", MAX_COLLECTIVE_PROMPT_CHARACTERS)
        text = f"User: {prompt}\nAssistant: {response}"
        if not self.nodes:
            candidate_corpus = f"{self.shared_corpus}\n{text}".strip()
            if len(candidate_corpus) > MAX_TRAINING_CHARACTERS:
                raise ValueError(
                    f"shared training corpus cannot exceed {MAX_TRAINING_CHARACTERS} characters"
                )
            self.shared_corpus = candidate_corpus
            self._save_metadata()
            self._record_run(
                "agent.teach",
                {
                    "source_type": "conversation",
                    "prompt_characters": len(prompt),
                    "response_characters": len(response),
                },
                {"trained_nodes": 0, "shared_training_characters": len(self.shared_corpus)},
                nodes=[],
            )
            return {"trained_nodes": 0, "training_characters": len(self.shared_corpus)}
        return self.train_all(
            text,
            epochs=epochs,
            learning_rate=learning_rate,
            source_type="conversation",
        )

    def think(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 48,
        run_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = _validated_text(prompt, "prompt", MAX_COLLECTIVE_PROMPT_CHARACTERS)
        if len(self.nodes) < 2:
            raise ValueError("collective responses require at least two agent nodes")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= MAX_AGENT_RESPONSE_TOKENS
        ):
            raise ValueError(
                f"max_new_tokens must be a whole number from 1 to {MAX_AGENT_RESPONSE_TOKENS}"
            )

        private_messages: list[dict[str, str]] = []
        shared_messages: list[dict[str, str]] = []
        episode: list[dict[str, str]] = []
        for node in self.nodes:
            if node.model.trained_characters == 0:
                raise RuntimeError(f"agent {node.name!r} has not been trained")
            context = f"Role: {node.role}\nTask: {prompt}\nResponse:"
            output = self._generate(node, context, max_new_tokens)
            private_messages.append(
                {"agent_id": node.agent_id, "name": node.name, "round": "private", "text": output}
            )
            episode.append({"agent_id": node.agent_id, "prompt": context, "completion": output})

        for node in self.nodes:
            peer_messages = [*private_messages, *shared_messages]
            peer_text = "\n".join(
                f"{item['name']}: {item['text']}" for item in peer_messages
            )
            context = (
                f"Role: {node.role}\n"
                f"Peer messages:\n{peer_text}\n"
                f"Task: {prompt}\n"
                "Revise the response using the peer messages:\n"
            )
            output = self._generate(node, context, max_new_tokens)
            shared_messages.append(
                {"agent_id": node.agent_id, "name": node.name, "round": "shared", "text": output}
            )
            episode.append({"agent_id": node.agent_id, "prompt": context, "completion": output})

        self._last_episode = episode
        result = {
            "response": shared_messages[-1]["text"],
            "messages": [*private_messages, *shared_messages],
            "communication": "two-round all-to-all shared context",
            "feedback_ready": True,
        }
        for node in self.nodes:
            self._save_node(node)
        self._record_run(
            "agent.collective_inference",
            {"prompt_characters": len(prompt), "max_new_tokens": max_new_tokens},
            {
                "message_count": len(episode),
                "message_character_counts": [len(item["completion"]) for item in episode],
                "final_response_characters": len(shared_messages[-1]["text"]),
                "run_details": run_details or {},
            },
        )
        return result

    def reinforce(
        self,
        reward: float,
        *,
        discount: float = 0.95,
        learning_rate: float = 0.01,
        reward_source: str = "user-rating",
        verifier_score: float | None = None,
        reward_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._last_episode is None:
            raise RuntimeError("run a collective response before submitting episode feedback")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not -1 <= reward <= 1
        ):
            raise ValueError("reward must be finite and between -1 and 1")
        if (
            isinstance(discount, bool)
            or not isinstance(discount, (int, float))
            or not math.isfinite(discount)
            or not 0 <= discount <= 1
        ):
            raise ValueError("discount must be finite and between 0 and 1")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 0.1
        ):
            raise ValueError("learning_rate must be finite and between 0 and 0.1")
        if not isinstance(reward_source, str) or not reward_source:
            raise ValueError("reward_source must be a non-empty string")
        if reward_details is not None and not isinstance(reward_details, dict):
            raise TypeError("reward_details must be a mapping")
        if verifier_score is not None and (
            isinstance(verifier_score, bool)
            or not isinstance(verifier_score, (int, float))
            or not math.isfinite(verifier_score)
            or not 0 <= verifier_score <= 1
        ):
            raise ValueError("verifier_score must be finite and between 0 and 1")

        returns = [
            float(reward) * float(discount) ** (len(self._last_episode) - index - 1)
            for index in range(len(self._last_episode))
        ]
        advantages = [max(-1.0, min(1.0, value - self.reward_baseline)) for value in returns]
        log_probabilities = []
        for item, advantage in zip(self._last_episode, advantages, strict=True):
            node = self._node_by_id(item["agent_id"])
            log_probabilities.append(
                node.model.reinforce(
                    item["prompt"],
                    item["completion"],
                    advantage,
                    learning_rate=learning_rate,
                )
            )
            self._save_node(node)

        self.feedback_count += 1
        self.reward_baseline = (
            float(reward)
            if self.feedback_count == 1
            else 0.9 * self.reward_baseline + 0.1 * float(reward)
        )
        self._save_metadata()
        self._last_episode = None
        result = {
            "updated_nodes": len(self.nodes),
            "trajectory_steps": len(advantages),
            "mean_log_probability": sum(log_probabilities) / len(log_probabilities),
            "reward": float(reward),
            "baseline": self.reward_baseline,
            "feedback_count": self.feedback_count,
            "algorithm": "episodic REINFORCE with discounted return baseline",
        }
        self._record_run(
            "agent.episodic_reinforce",
            {
                "reward": float(reward),
                "discount": float(discount),
                "learning_rate": float(learning_rate),
                "trajectory_steps": len(advantages),
                "reward_source": reward_source,
                "verifier_score": verifier_score,
                "reward_details": reward_details or {},
            },
            {
                "updated_nodes": len(self.nodes),
                "reward_baseline": self.reward_baseline,
                "mean_log_probability": result["mean_log_probability"],
                "verifier_score": verifier_score,
            },
        )
        return result

    def _generate(self, node: AgentNode, prompt: str, max_new_tokens: int) -> str:
        output = node.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=min(16, max(1, len(node.model.vocabulary))),
        )
        if not output:
            raise RuntimeError(f"agent {node.name!r} generated an empty response")
        return output

    def _node_by_id(self, agent_id: str) -> AgentNode:
        for node in self.nodes:
            if node.agent_id == agent_id:
                return node
        raise ValueError(f"episode references unknown agent {agent_id!r}")

    def _save_node(self, node: AgentNode) -> None:
        node.model.save(self.directory / f"{node.agent_id}.npz")

    def _save_metadata(self) -> None:
        data = {
            "version": 1,
            "shared_corpus": self.shared_corpus,
            "reward_baseline": self.reward_baseline,
            "feedback_count": self.feedback_count,
            "nodes": [
                {"id": node.agent_id, "name": node.name, "role": node.role, "seed": node.seed}
                for node in self.nodes
            ],
        }
        _atomic_write(
            self.metadata_path,
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        )

    def _load(self) -> None:
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported agent-network file version")
            corpus = data.get("shared_corpus")
            raw_nodes = data.get("nodes")
            baseline = data.get("reward_baseline")
            count = data.get("feedback_count")
            if (
                not isinstance(corpus, str)
                or len(corpus) > MAX_TRAINING_CHARACTERS
                or not isinstance(raw_nodes, list)
                or len(raw_nodes) > MAX_AGENT_NODES
                or isinstance(baseline, bool)
                or not isinstance(baseline, (int, float))
                or not math.isfinite(baseline)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("agent-network metadata has invalid fields")
            nodes = []
            seen = set()
            for item in raw_nodes:
                if (
                    not isinstance(item, dict)
                    or not {"id", "name", "role"} <= set(item)
                    or set(item) - {"id", "name", "role", "seed"}
                ):
                    raise ValueError("agent-node metadata has invalid fields")
                agent_id = item["id"]
                if not isinstance(agent_id, str) or not re.fullmatch(r"[a-f0-9]{32}", agent_id):
                    raise ValueError("agent-node identifier is invalid")
                if agent_id in seen:
                    raise ValueError("agent-node identifiers must be unique")
                seen.add(agent_id)
                name = _validated_text(item["name"], "name", MAX_AGENT_NAME_CHARACTERS)
                role = _validated_text(item["role"], "role", MAX_AGENT_ROLE_CHARACTERS)
                model = CharacterLanguageModel.load(self.directory / f"{agent_id}.npz")
                node_seed = item.get("seed", self._seed + len(nodes))
                if isinstance(node_seed, bool) or not isinstance(node_seed, int):
                    raise TypeError("agent-node seed must be an integer")
                nodes.append(AgentNode(agent_id, name, role, node_seed, model))
            self.shared_corpus = corpus
            self.reward_baseline = float(baseline)
            self.feedback_count = count
            self.nodes = nodes
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"could not load agent network: {exc}") from exc

    def _record_run(
        self,
        run_type: str,
        settings: dict[str, Any],
        results: dict[str, Any],
        *,
        nodes: list[AgentNode] | None = None,
    ) -> None:
        selected_nodes = self.nodes if nodes is None else nodes
        timestamp = datetime.now(UTC)
        run_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
        code_files = {
            "agent_network.py": Path(__file__),
            "language_model.py": Path(__file__).with_name("language_model.py"),
            "server.py": Path(__file__).with_name("server.py"),
            "reasoning_data.py": Path(__file__).with_name("reasoning_data.py"),
        }
        record = {
            "run_id": run_id,
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "run_type": run_type,
            "agent_seeds": {node.agent_id: node.seed for node in selected_nodes},
            "settings": settings,
            "results": results,
            "code_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in code_files.items()
            },
        }
        self.run_records_directory.mkdir(parents=True, exist_ok=True)
        target = self.run_records_directory / f"{run_id}.json"
        _atomic_write(
            target,
            json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"),
        )


def _validated_text(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} must not exceed {maximum} characters")
    return normalized


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
