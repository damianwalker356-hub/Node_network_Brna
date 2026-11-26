"""Bounded latent messaging, differentiable graph gates, and episodic policy RL."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brna.agent_network import AgentNetwork, AgentNode
from brna.proto import Adam, sigmoid
from brna.reasoning_data import generate_examples

MAX_GRAPH_NODES = 8
MAX_LATENT_SIZE = 128
MAX_EPISODE_STEPS = 256
MAX_CONTEXT_CHARACTERS = 4_000
MAX_NODE_RESPONSE_TOKENS = 512


def _bounded_vector(values: np.ndarray, size: int, max_norm: float, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    norm = float(np.linalg.norm(vector))
    if norm > max_norm + 1e-8:
        raise ValueError(f"{name} norm must not exceed {max_norm}")
    return vector


def _clip_norm(values: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values if norm <= max_norm or norm == 0 else values * (max_norm / norm)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


@dataclass(frozen=True)
class LatentAgentMessage:
    agent_id: str
    name: str
    role: str
    text: str
    latent: np.ndarray
    training_prompt: str
    soft_prefix: Any | None = None
    context_states: Any | None = None


class BoundedLatentNode:
    """Adapts one trained user-created language agent to bounded vector messages."""

    def __init__(
        self,
        agent: AgentNode,
        *,
        latent_size: int = 8,
        max_norm: float = 1.0,
        hidden_size: int | None = None,
        context_encoder: Callable[[AgentNode, str], np.ndarray] | None = None,
        response_generator: Callable[[AgentNode, str, int], str] | None = None,
    ) -> None:
        if not isinstance(agent, AgentNode):
            raise TypeError("agent must be an AgentNode")
        if (
            isinstance(latent_size, bool)
            or not isinstance(latent_size, int)
            or not 1 <= latent_size <= MAX_LATENT_SIZE
        ):
            raise ValueError(f"latent_size must be from 1 to {MAX_LATENT_SIZE}")
        if (
            isinstance(max_norm, bool)
            or not isinstance(max_norm, (int, float))
            or not math.isfinite(max_norm)
            or max_norm <= 0
        ):
            raise ValueError("max_norm must be finite and positive")
        self.agent = agent
        self.latent_size = latent_size
        self.max_norm = float(max_norm)
        self.context_encoder = context_encoder
        self.response_generator = response_generator
        projection_input_size = (
            agent.model.hidden_size if hidden_size is None else hidden_size
        )
        if (
            isinstance(projection_input_size, bool)
            or not isinstance(projection_input_size, int)
            or projection_input_size < 1
        ):
            raise ValueError("hidden_size must be a positive whole number")
        digest = hashlib.sha256(agent.agent_id.encode("utf-8")).digest()
        projection_seed = (agent.seed ^ int.from_bytes(digest[:4], "little")) % 2**32
        rng = np.random.default_rng(projection_seed)
        self.projection = rng.normal(
            0.0,
            1.0 / np.sqrt(projection_input_size),
            (projection_input_size, latent_size),
        )

    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(
                f"text must not exceed {MAX_CONTEXT_CHARACTERS} characters"
            )
        if (
            self.context_encoder is None
            and self.agent.model.trained_characters == 0
        ):
            raise RuntimeError(f"agent {self.agent.name!r} has not been trained")
        context = f"Role: {self.agent.role}\n{text}"
        hidden = (
            self.agent.model.encode_context(context)
            if self.context_encoder is None
            else self.context_encoder(self.agent, context)
        )
        hidden = np.asarray(hidden, dtype=float)
        if hidden.shape != (self.projection.shape[0],) or not np.isfinite(hidden).all():
            raise RuntimeError("agent encoder returned an invalid hidden state")
        return _clip_norm(np.tanh(hidden @ self.projection), self.max_norm)

    def respond(
        self,
        prompt: str,
        incoming_latent: np.ndarray,
        *,
        max_new_tokens: int = 32,
    ) -> LatentAgentMessage:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(
                f"prompt must not exceed {MAX_CONTEXT_CHARACTERS} characters"
            )
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= MAX_NODE_RESPONSE_TOKENS
        ):
            raise ValueError(
                f"max_new_tokens must be from 1 to {MAX_NODE_RESPONSE_TOKENS}"
            )
        latent = _bounded_vector(
            incoming_latent, self.latent_size, self.max_norm, "incoming_latent"
        )
        latent_text = np.array2string(
            latent,
            precision=4,
            separator=", ",
            max_line_width=100_000,
        )
        context = (
            f"Role: {self.agent.role}\n"
            f"Task: {prompt}\n"
            f"Bounded peer latent: {latent_text}\n"
            "Response:"
        )
        response = (
            self.agent.model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=min(16, max(1, len(self.agent.model.vocabulary))),
            )
            if self.response_generator is None
            else self.response_generator(self.agent, context, max_new_tokens)
        )
        if not response:
            raise RuntimeError(f"agent {self.agent.name!r} generated an empty response")
        output_latent = self.encode_text(response)
        return LatentAgentMessage(
            self.agent.agent_id,
            self.agent.name,
            self.agent.role,
            response,
            output_latent,
            context,
        )


@dataclass
class GatedGraphOutput:
    action_logits: np.ndarray
    node_messages: np.ndarray
    states: list[np.ndarray]
    aggregates: list[np.ndarray]
    gate_matrix: np.ndarray


class DifferentiableGatedGraph:
    """Trainable directed graph with bounded latent messages and edge gates."""

    def __init__(
        self,
        *,
        node_count: int,
        latent_size: int,
        action_count: int,
        hidden_size: int | None = None,
        hops: int = 3,
        readout_node: int | None = None,
        adjacency: np.ndarray | None = None,
        max_norm: float = 1.0,
        learning_rate: float = 0.01,
        seed: int = 0,
    ) -> None:
        for name, value, minimum, maximum in (
            ("node_count", node_count, 1, MAX_GRAPH_NODES),
            ("latent_size", latent_size, 1, MAX_LATENT_SIZE),
            ("action_count", action_count, 2, 256),
            ("hops", hops, 0, 16),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be from {minimum} to {maximum}")
        hidden_size = latent_size if hidden_size is None else hidden_size
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or not 1 <= hidden_size <= MAX_LATENT_SIZE
        ):
            raise ValueError(f"hidden_size must be from 1 to {MAX_LATENT_SIZE}")
        if (
            isinstance(max_norm, bool)
            or not isinstance(max_norm, (int, float))
            or not math.isfinite(max_norm)
            or max_norm <= 0
        ):
            raise ValueError("max_norm must be finite and positive")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1
        ):
            raise ValueError("learning_rate must be finite and in (0, 1]")
        readout_node = node_count - 1 if readout_node is None else readout_node
        if (
            isinstance(readout_node, bool)
            or not isinstance(readout_node, int)
            or not 0 <= readout_node < node_count
        ):
            raise ValueError("readout_node must refer to an existing graph node")

        self.node_count = node_count
        self.latent_size = latent_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.hops = hops
        self.readout_node = readout_node
        self.max_norm = float(max_norm)
        self.learning_rate = float(learning_rate)
        rng = np.random.default_rng(seed)
        if adjacency is None:
            graph = np.zeros((node_count, node_count), dtype=float)
            if node_count > 1:
                for target in range(node_count):
                    graph[target, (target - 1) % node_count] = 1.0
                extra_edges = min(2, node_count - 1)
                for target in range(node_count):
                    candidates = [
                        source
                        for source in range(node_count)
                        if source != target and graph[target, source] == 0
                    ]
                    count = min(extra_edges - 1, len(candidates))
                    if count:
                        chosen = rng.choice(candidates, count, replace=False)
                        graph[target, chosen] = 1.0
            adjacency = graph
        graph = np.asarray(adjacency, dtype=float)
        if (
            graph.shape != (node_count, node_count)
            or not np.isfinite(graph).all()
            or not np.isin(graph, (0.0, 1.0)).all()
            or np.any(np.diag(graph))
        ):
            raise ValueError("adjacency must be a binary square matrix without self-edges")
        self.adjacency = graph.copy()
        self.edge_sources, self.edge_targets = np.nonzero(self.adjacency)

        def gaussian(*shape: int) -> np.ndarray:
            return rng.normal(0.0, 1.0 / np.sqrt(shape[0]), shape)

        self.input_weights = gaussian(latent_size, hidden_size)
        self.node_embeddings = rng.normal(
            0.0, 0.1, (node_count, hidden_size)
        )
        self.self_weights = gaussian(hidden_size, hidden_size)
        self.message_weights = gaussian(hidden_size, hidden_size)
        self.hidden_bias = np.zeros(hidden_size)
        self.policy_weights = gaussian(hidden_size, action_count)
        self.policy_bias = np.zeros(action_count)
        self.edge_gates = np.zeros(len(self.edge_sources))
        self.parameters = [
            self.input_weights,
            self.node_embeddings,
            self.self_weights,
            self.message_weights,
            self.hidden_bias,
            self.policy_weights,
            self.policy_bias,
            self.edge_gates,
        ]
        self.optimizer = Adam(self.parameters, lr=self.learning_rate)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self.parameters)

    def gate_matrix(self) -> np.ndarray:
        matrix = np.zeros_like(self.adjacency)
        matrix[self.edge_sources, self.edge_targets] = sigmoid(self.edge_gates)
        return matrix

    def forward(self, latents: np.ndarray) -> GatedGraphOutput:
        values = np.asarray(latents, dtype=float)
        if (
            values.ndim != 3
            or values.shape[1:] != (self.node_count, self.latent_size)
            or values.shape[0] == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"latents must have shape (batch, {self.node_count}, {self.latent_size})"
            )
        norms = np.linalg.norm(values, axis=-1)
        if np.any(norms > self.max_norm + 1e-8):
            raise ValueError(f"each input latent norm must not exceed {self.max_norm}")
        gates = self.gate_matrix()
        scaled_gates = gates / max(1, self.node_count)
        states = [
            np.tanh(values @ self.input_weights + self.node_embeddings)
            / np.sqrt(self.hidden_size)
        ]
        messages = []
        aggregates = []
        for _ in range(self.hops):
            message = states[-1]
            aggregate = np.einsum("ij,bjd->bid", scaled_gates, message)
            next_state = np.tanh(
                states[-1] @ self.self_weights
                + aggregate @ self.message_weights
                + self.hidden_bias
            ) / np.sqrt(self.hidden_size)
            messages.append(message)
            aggregates.append(aggregate)
            states.append(next_state)
        logits = states[-1][:, self.readout_node] @ self.policy_weights + self.policy_bias
        return GatedGraphOutput(
            logits,
            states[-1],
            states,
            aggregates,
            gates,
        )

    def train_supervised(self, latents: np.ndarray, actions: np.ndarray) -> float:
        values = np.asarray(latents, dtype=float)
        targets = np.asarray(actions)
        output = self.forward(values)
        if (
            targets.shape != (len(values),)
            or not np.issubdtype(targets.dtype, np.integer)
            or np.any(targets < 0)
            or np.any(targets >= self.action_count)
        ):
            raise ValueError("actions must be integer class indices for each batch item")
        probabilities = np.stack([_softmax(row) for row in output.action_logits])
        losses = -np.log(
            np.maximum(probabilities[np.arange(len(values)), targets], 1e-12)
        )
        logits_gradient = probabilities
        logits_gradient[np.arange(len(values)), targets] -= 1.0
        logits_gradient /= len(values)
        self._apply_gradients(values, output, logits_gradient, self.learning_rate)
        return float(np.mean(losses))

    def policy_gradient_step(
        self,
        latents: np.ndarray,
        action: int,
        advantage: float,
        *,
        learning_rate: float,
    ) -> float:
        values = np.asarray(latents, dtype=float)
        output = self.forward(values)
        if (
            len(values) != 1
            or isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < self.action_count
        ):
            raise ValueError("policy updates require one latent state and a valid action")
        if (
            isinstance(advantage, bool)
            or not isinstance(advantage, (int, float))
            or not math.isfinite(advantage)
            or not -1 <= advantage <= 1
        ):
            raise ValueError("advantage must be finite and between -1 and 1")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 0.1
        ):
            raise ValueError("learning_rate must be finite and in (0, 0.1]")
        probabilities = _softmax(output.action_logits[0])
        logits_gradient = np.zeros_like(output.action_logits)
        logits_gradient[0] = probabilities
        logits_gradient[0, action] -= 1.0
        logits_gradient *= float(advantage)
        loss = -float(advantage) * float(np.log(max(probabilities[action], 1e-12)))
        self._apply_gradients(
            values,
            output,
            logits_gradient,
            float(learning_rate),
        )
        return loss

    def _apply_gradients(
        self,
        inputs: np.ndarray,
        output: GatedGraphOutput,
        logits_gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        final_state = output.states[-1][:, self.readout_node]
        policy_weight_gradient = final_state.T @ logits_gradient
        policy_bias_gradient = logits_gradient.sum(axis=0)
        state_gradient = np.zeros_like(output.states[-1])
        state_gradient[:, self.readout_node] = (
            logits_gradient @ self.policy_weights.T
        )
        self_weight_gradient = np.zeros_like(self.self_weights)
        message_weight_gradient = np.zeros_like(self.message_weights)
        bias_gradient = np.zeros_like(self.hidden_bias)
        gate_matrix_gradient = np.zeros_like(self.adjacency)
        scaled_gates = output.gate_matrix / max(1, self.node_count)
        hidden_scale = np.sqrt(self.hidden_size)

        for hop in range(self.hops, 0, -1):
            current_state = output.states[hop]
            preactivation_gradient = (
                state_gradient
                * (1.0 - self.hidden_size * current_state**2)
                / hidden_scale
            )
            previous_state = output.states[hop - 1]
            aggregate = output.aggregates[hop - 1]
            message = output.states[hop - 1]
            bias_gradient += preactivation_gradient.sum(axis=(0, 1))
            self_weight_gradient += np.einsum(
                "bnd,bne->de", previous_state, preactivation_gradient
            )
            message_weight_gradient += np.einsum(
                "bnd,bne->de", aggregate, preactivation_gradient
            )
            aggregate_gradient = preactivation_gradient @ self.message_weights.T
            gate_matrix_gradient += (
                np.einsum("bid,bjd->ij", aggregate_gradient, message)
                / max(1, self.node_count)
            )
            state_gradient = (
                preactivation_gradient @ self.self_weights.T
                + np.einsum("ij,bid->bjd", scaled_gates, aggregate_gradient)
            )

        initial_state = output.states[0]
        initial_gradient = (
            state_gradient
            * (1.0 - self.hidden_size * initial_state**2)
            / hidden_scale
        )
        input_weight_gradient = np.einsum(
            "bnl,bnh->lh", inputs, initial_gradient
        )
        embedding_gradient = initial_gradient.sum(axis=0)
        gate_values = sigmoid(self.edge_gates)
        edge_gradient = (
            gate_matrix_gradient[self.edge_sources, self.edge_targets]
            * gate_values
            * (1.0 - gate_values)
        )
        gradients = [
            input_weight_gradient,
            embedding_gradient,
            self_weight_gradient,
            message_weight_gradient,
            bias_gradient,
            policy_weight_gradient,
            policy_bias_gradient,
            edge_gradient,
        ]
        original_rate = self.optimizer.lr
        self.optimizer.lr = learning_rate
        self.optimizer.step(gradients)
        self.optimizer.lr = original_rate

    def save(self, path: str | Path) -> None:
        metadata = {
            "version": 1,
            "node_count": self.node_count,
            "latent_size": self.latent_size,
            "action_count": self.action_count,
            "hidden_size": self.hidden_size,
            "hops": self.hops,
            "readout_node": self.readout_node,
            "max_norm": self.max_norm,
            "learning_rate": self.learning_rate,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata)),
                adjacency=self.adjacency,
                input_weights=self.input_weights,
                node_embeddings=self.node_embeddings,
                self_weights=self.self_weights,
                message_weights=self.message_weights,
                hidden_bias=self.hidden_bias,
                policy_weights=self.policy_weights,
                policy_bias=self.policy_bias,
                edge_gates=self.edge_gates,
            )

    @classmethod
    def load(cls, path: str | Path) -> DifferentiableGatedGraph:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                if metadata.get("version") != 1:
                    raise ValueError("unsupported gated graph version")
                graph = cls(
                    node_count=metadata["node_count"],
                    latent_size=metadata["latent_size"],
                    action_count=metadata["action_count"],
                    hidden_size=metadata["hidden_size"],
                    hops=metadata["hops"],
                    readout_node=metadata["readout_node"],
                    adjacency=data["adjacency"],
                    max_norm=metadata["max_norm"],
                    learning_rate=metadata["learning_rate"],
                    seed=0,
                )
                stored_parameters = (
                    data["input_weights"],
                    data["node_embeddings"],
                    data["self_weights"],
                    data["message_weights"],
                    data["hidden_bias"],
                    data["policy_weights"],
                    data["policy_bias"],
                    data["edge_gates"],
                )
                for parameter, stored in zip(
                    graph.parameters, stored_parameters, strict=True
                ):
                    if parameter.shape != stored.shape or not np.isfinite(stored).all():
                        raise ValueError("gated graph parameters have invalid shape or values")
                    parameter[...] = stored
                return graph
        except (OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read gated graph: {exc}") from exc


class DifferentiableGraphGateTrainer:
    """Supervised task trainer for the graph transforms and directed edge gates."""

    def __init__(self, graph: DifferentiableGatedGraph) -> None:
        if not isinstance(graph, DifferentiableGatedGraph):
            raise TypeError("graph must be a DifferentiableGatedGraph")
        self.graph = graph
        self.update_count = 0

    def train_step(self, latents: np.ndarray, actions: np.ndarray) -> float:
        loss = self.graph.train_supervised(latents, actions)
        self.update_count += 1
        return loss

    def fit(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int = 1,
    ) -> dict[str, Any]:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 1_000:
            raise ValueError("epochs must be from 1 to 1000")
        losses = [self.train_step(latents, actions) for _ in range(epochs)]
        return {
            "updates": self.update_count,
            "losses": losses,
            "mean_gate": float(np.mean(sigmoid(self.graph.edge_gates)))
            if self.graph.edge_gates.size
            else 0.0,
        }


class ClosedLoopRLPolicyOptimizer:
    """Episodic discounted REINFORCE optimizer for graph-selected actions."""

    def __init__(
        self,
        graph: DifferentiableGatedGraph,
        *,
        discount: float = 0.95,
        baseline_rate: float = 0.1,
        learning_rate: float = 0.01,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        seed: int = 0,
    ) -> None:
        if not isinstance(graph, DifferentiableGatedGraph):
            raise TypeError("graph must be a DifferentiableGatedGraph")
        for name, value in (("discount", discount), ("baseline_rate", baseline_rate)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 0.1
        ):
            raise ValueError("learning_rate must be finite and in (0, 0.1]")
        if (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or not 1 <= max_episode_steps <= MAX_EPISODE_STEPS
        ):
            raise ValueError(
                f"max_episode_steps must be from 1 to {MAX_EPISODE_STEPS}"
            )
        self.graph = graph
        self.discount = float(discount)
        self.baseline_rate = float(baseline_rate)
        self.learning_rate = float(learning_rate)
        self.max_episode_steps = max_episode_steps
        self.baseline = 0.0
        self.episode_count = 0
        self.rng = np.random.default_rng(seed)
        self._episode: list[dict[str, Any]] | None = None

    @property
    def episode_active(self) -> bool:
        return self._episode is not None

    @property
    def awaiting_reward(self) -> bool:
        return bool(
            self._episode
            and self._episode[-1]["reward"] is None
        )

    def begin_episode(self) -> None:
        if self._episode is not None:
            raise RuntimeError("an episode is already active")
        self._episode = []

    def choose_action(self, latents: np.ndarray) -> dict[str, Any]:
        if self._episode is None:
            raise RuntimeError("begin an episode before choosing an action")
        if self._episode and self._episode[-1]["reward"] is None:
            raise RuntimeError("record the previous action reward first")
        if len(self._episode) >= self.max_episode_steps:
            raise RuntimeError("episode reached its configured step limit")
        values = np.asarray(latents, dtype=float)
        if values.ndim == 2:
            values = values[None, ...]
        if values.shape[0] != 1:
            raise ValueError("policy action selection requires one graph state")
        output = self.graph.forward(values)
        probabilities = _softmax(output.action_logits[0])
        action = int(self.rng.choice(self.graph.action_count, p=probabilities))
        self._episode.append(
            {
                "latents": values.copy(),
                "action": action,
                "reward": None,
            }
        )
        return {
            "action": action,
            "probabilities": probabilities,
            "policy": "categorical-gated-graph",
        }

    def observe_reward(
        self,
        reward: float,
        *,
        done: bool = False,
    ) -> dict[str, Any] | None:
        if self._episode is None or not self._episode:
            raise RuntimeError("choose an action before recording its reward")
        if self._episode[-1]["reward"] is not None:
            raise RuntimeError("the current action already has a reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not -1 <= reward <= 1
        ):
            raise ValueError("reward must be finite and between -1 and 1")
        if not isinstance(done, bool):
            raise TypeError("done must be a boolean")
        self._episode[-1]["reward"] = float(reward)
        return self.finish_episode() if done else None

    def finish_episode(self) -> dict[str, Any]:
        if self._episode is None or not self._episode:
            raise RuntimeError("cannot finish an empty or inactive episode")
        if any(step["reward"] is None for step in self._episode):
            raise RuntimeError("every chosen action must have a reward before finishing")
        returns = [0.0] * len(self._episode)
        running_return = 0.0
        for index in range(len(self._episode) - 1, -1, -1):
            running_return = (
                float(self._episode[index]["reward"])
                + self.discount * running_return
            )
            returns[index] = running_return
        advantages = [
            float(np.clip(value - self.baseline, -1.0, 1.0))
            for value in returns
        ]
        losses = []
        for step, advantage in zip(self._episode, advantages, strict=True):
            losses.append(
                self.graph.policy_gradient_step(
                    step["latents"],
                    step["action"],
                    advantage,
                    learning_rate=self.learning_rate,
                )
            )
        mean_return = float(np.mean(returns))
        self.baseline = (
            mean_return
            if self.episode_count == 0
            else (1.0 - self.baseline_rate) * self.baseline
            + self.baseline_rate * mean_return
        )
        self.episode_count += 1
        result = {
            "updated_steps": len(self._episode),
            "mean_return": mean_return,
            "mean_advantage": float(np.mean(advantages)),
            "advantages": advantages,
            "mean_policy_loss": float(np.mean(losses)),
            "baseline": self.baseline,
            "episodes": self.episode_count,
            "algorithm": "discounted episodic REINFORCE with running return baseline",
        }
        self._episode = None
        return result


class AgentGraphBridge:
    """Connects persisted user-created text agents to the latent graph and policy."""

    def __init__(
        self,
        agents: AgentNetwork,
        *,
        latent_size: int = 8,
        action_count: int = 4,
        hops: int = 3,
        seed: int = 0,
        agent_learning_rate: float = 0.01,
    ) -> None:
        if not isinstance(agents, AgentNetwork):
            raise TypeError("agents must be an AgentNetwork")
        if not 1 <= len(agents.nodes) <= MAX_GRAPH_NODES:
            raise ValueError(f"create between 1 and {MAX_GRAPH_NODES} user agents first")
        self.agents = agents
        self.nodes = agents.bounded_latent_nodes(latent_size=latent_size)
        self.graph = DifferentiableGatedGraph(
            node_count=len(self.nodes),
            latent_size=latent_size,
            action_count=action_count,
            hops=hops,
            seed=seed,
        )
        self.trainer = DifferentiableGraphGateTrainer(self.graph)
        self.policy = ClosedLoopRLPolicyOptimizer(self.graph, seed=seed + 1)
        self.soft_prefix_graph = (
            agents.backbone.create_soft_prefix_graph(
                node_count=len(self.nodes),
                adjacency=self.graph.adjacency,
                latent_size=latent_size,
                hops=hops,
                seed=seed + 2,
            )
            if agents.backbone is not None
            else None
        )
        self._last_context_states = None
        self.last_policy_latents: np.ndarray | None = None
        if (
            isinstance(agent_learning_rate, bool)
            or not isinstance(agent_learning_rate, (int, float))
            or not math.isfinite(agent_learning_rate)
            or not 0 < agent_learning_rate <= 0.1
        ):
            raise ValueError("agent_learning_rate must be finite and in (0, 0.1]")
        self.agent_learning_rate = float(agent_learning_rate)
        self._last_messages: list[LatentAgentMessage] = []
        self._episode_messages: list[list[LatentAgentMessage]] = []

    def think(self, prompt: str, *, max_new_tokens: int = 32) -> dict[str, Any]:
        for node in self.agents.nodes:
            is_trained = (
                node.model.trained_characters > 0
                if self.agents.backbone is None
                else node.backbone_trained_characters > 0
            )
            if not is_trained:
                raise RuntimeError(f"agent {node.name!r} has not been trained")
        if self.soft_prefix_graph is not None:
            return self._think_with_soft_prefix(prompt, max_new_tokens)
        initial_latents = np.stack(
            [node.encode_text(prompt) for node in self.nodes]
        )[None, ...]
        initial_graph = self.graph.forward(initial_latents)
        incoming = initial_graph.node_messages[0]
        messages = [
            node.respond(
                prompt,
                incoming[index],
                max_new_tokens=max_new_tokens,
            )
            for index, node in enumerate(self.nodes)
        ]
        self.agents.save_models()
        response_latents = np.stack([message.latent for message in messages])[None, ...]
        final_graph = self.graph.forward(response_latents)
        self.last_policy_latents = response_latents
        self._last_messages = messages
        return {
            "response": messages[-1].text,
            "messages": [
                {
                    "agent_id": message.agent_id,
                    "name": message.name,
                    "role": message.role,
                    "text": message.text,
                    "latent": message.latent.tolist(),
                }
                for message in messages
            ],
            "action_logits": final_graph.action_logits[0].tolist(),
            "communication": "bounded latent graph pass followed by role-agent responses",
        }

    def _think_with_soft_prefix(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        backbone = self.agents.backbone
        contexts = [
            f"Role: {node.agent.role}\nTask: {prompt}\nResponse:"
            for node in self.nodes
        ]
        context_states = backbone.stack_contexts(
            [
                backbone.encode_context_tensor(
                    context,
                    adapter_id=node.agent.agent_id,
                    role=node.agent.role,
                )
                for node, context in zip(self.nodes, contexts, strict=True)
            ]
        )
        graph_output = self.soft_prefix_graph(context_states)
        prefixes = graph_output["prefix_embeddings"]
        response_latents = backbone.tensor_to_numpy(graph_output["latents"])
        messages = []
        for index, (node, context) in enumerate(
            zip(self.nodes, contexts, strict=True)
        ):
            prefix = prefixes[index : index + 1]
            response = backbone.generate(
                context,
                adapter_id=node.agent.agent_id,
                role=node.agent.role,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                soft_prefix=prefix,
            )
            if not response:
                raise RuntimeError(f"agent {node.agent.name!r} generated an empty response")
            messages.append(
                LatentAgentMessage(
                    node.agent.agent_id,
                    node.agent.name,
                    node.agent.role,
                    response,
                    response_latents[index],
                    context,
                    prefix,
                    context_states,
                )
            )
        self._last_context_states = context_states
        self._last_messages = messages
        self.last_policy_latents = response_latents[None, ...]
        action_logits = self.graph.forward(self.last_policy_latents).action_logits[0]
        return {
            "response": messages[-1].text,
            "messages": [
                {
                    "agent_id": message.agent_id,
                    "name": message.name,
                    "role": message.role,
                    "text": message.text,
                    "latent": message.latent.tolist(),
                }
                for message in messages
            ],
            "action_logits": action_logits.tolist(),
            "communication": "differentiable graph soft-prefix embeddings",
        }

    def grpo_reasoning_episode(
        self,
        task_name: str,
        *,
        seed: int = 0,
        index: int = 0,
        group_size: int = 4,
        max_new_tokens: int = 128,
        kl_beta: float = 0.02,
        learning_rate: float = 1e-5,
    ) -> dict[str, Any]:
        if self.soft_prefix_graph is None:
            raise RuntimeError(
                "GRPO requires the optional local pretrained causal-LM backend"
            )
        if (
            isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or not 2 <= group_size <= 8
        ):
            raise ValueError("group_size must be from 2 to 8")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= MAX_NODE_RESPONSE_TOKENS
        ):
            raise ValueError(
                f"max_new_tokens must be from 1 to {MAX_NODE_RESPONSE_TOKENS}"
            )
        examples, dataset = generate_examples(task_name, size=index + 1, seed=seed)
        example = examples[index]
        backbone = self.agents.backbone
        prompts = [
            f"Role: {node.agent.role}\nTask: {example['question']}\nResponse:"
            for node in self.nodes
        ]
        context_states = backbone.stack_contexts(
            [
                backbone.encode_context_tensor(
                    prompt,
                    adapter_id=node.agent.agent_id,
                    role=node.agent.role,
                )
                for node, prompt in zip(self.nodes, prompts, strict=True)
            ]
        )
        reports = []
        for index_node, (node, prompt) in enumerate(
            zip(self.nodes, prompts, strict=True)
        ):
            sampled_graph = self.soft_prefix_graph(context_states)
            sample_prefix = sampled_graph["prefix_embeddings"][
                index_node : index_node + 1
            ]
            completions = [
                backbone.generate(
                    prompt,
                    adapter_id=node.agent.agent_id,
                    role=node.agent.role,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    soft_prefix=sample_prefix,
                )
                for _ in range(group_size)
            ]
            if any(not completion for completion in completions):
                raise RuntimeError(
                    f"agent {node.agent.name!r} produced an empty GRPO candidate"
                )
            rewards = []
            for completion in completions:
                score = dataset.score_answer(answer=completion, entry=example)
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                ):
                    raise RuntimeError("Reasoning Gym returned an invalid answer score")
                rewards.append(float(np.clip(score, 0.0, 1.0)))
            train_graph = self.soft_prefix_graph(context_states)
            train_prefix = train_graph["prefix_embeddings"][
                index_node : index_node + 1
            ]
            report = backbone.grpo_group_update(
                node.agent.agent_id,
                node.agent.role,
                prompt,
                completions,
                rewards,
                soft_prefix=train_prefix,
                graph_parameters=list(self.soft_prefix_graph.parameters()),
                learning_rate=learning_rate,
                kl_beta=kl_beta,
            )
            node.agent.backbone_reinforcement_updates += 1
            self.agents._save_node(node.agent)
            reports.append(
                {
                    "agent_id": node.agent.agent_id,
                    "name": node.agent.name,
                    **report,
                }
            )
        self._last_context_states = context_states
        return {
            "task_name": task_name,
            "seed": seed,
            "index": index,
            "group_size": group_size,
            "verified_candidates": len(self.nodes) * group_size,
            "mean_reward": float(
                np.mean([item["mean_reward"] for item in reports])
            ),
            "mean_kl": float(np.mean([item["mean_kl"] for item in reports])),
            "agents": reports,
            "algorithm": "group-relative policy optimization with frozen-base KL",
        }

    def begin_episode(self) -> None:
        self.last_policy_latents = None
        self._last_messages = []
        self._episode_messages = []
        self.policy.begin_episode()

    def choose_action(self) -> dict[str, Any]:
        if self.last_policy_latents is None or not self._last_messages:
            raise RuntimeError("run a latent-agent turn before choosing an action")
        action = self.policy.choose_action(self.last_policy_latents)
        self._episode_messages.append(list(self._last_messages))
        return action

    def observe_reward(self, reward: float, *, done: bool = False) -> dict[str, Any] | None:
        result = self.policy.observe_reward(reward, done=done)
        if result is None:
            return None
        if len(self._episode_messages) != result["updated_steps"]:
            raise RuntimeError("agent-message history does not match the policy episode")
        for messages, advantage in zip(
            self._episode_messages, result["advantages"], strict=True
        ):
            for index, message in enumerate(messages):
                agent = next(
                    node
                    for node in self.agents.nodes
                    if node.agent_id == message.agent_id
                )
                self.agents.reinforce_node(
                    agent,
                    message.training_prompt,
                    message.text,
                    advantage,
                    learning_rate=self.agent_learning_rate,
                    soft_prefix=(
                        self.soft_prefix_graph(message.context_states)[
                            "prefix_embeddings"
                        ][index : index + 1]
                        if self.soft_prefix_graph is not None
                        else None
                    ),
                    graph_parameters=(
                        list(self.soft_prefix_graph.parameters())
                        if self.soft_prefix_graph is not None
                        else ()
                    ),
                )
        self.agents.save_models()
        self._episode_messages = []
        return result

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.graph.save(target)
        if self.soft_prefix_graph is not None:
            prefix_path = target.with_suffix(target.suffix + ".prefix.npz")
            state = self.soft_prefix_graph.state_dict()
            with prefix_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    **{
                        name: value.detach().cpu().numpy()
                        for name, value in state.items()
                    },
                )
        metadata = {
            "version": 1,
            "agent_ids": [node.agent.agent_id for node in self.nodes],
            "discount": self.policy.discount,
            "baseline_rate": self.policy.baseline_rate,
            "learning_rate": self.policy.learning_rate,
            "max_episode_steps": self.policy.max_episode_steps,
            "baseline": self.policy.baseline,
            "episode_count": self.policy.episode_count,
            "soft_prefix_enabled": self.soft_prefix_graph is not None,
        }
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps(metadata, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        metadata_path = target.with_suffix(target.suffix + ".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read agent graph metadata: {exc}") from exc
        expected_agent_ids = [node.agent.agent_id for node in self.nodes]
        required_fields = {
            "version",
            "agent_ids",
            "discount",
            "baseline_rate",
            "learning_rate",
            "max_episode_steps",
            "baseline",
            "episode_count",
        }
        if (
            not isinstance(metadata, dict)
            or not required_fields.issubset(metadata)
            or metadata.get("version") != 1
            or metadata.get("agent_ids") != expected_agent_ids
        ):
            raise ValueError("agent graph checkpoint does not match the current agent nodes")
        prefix_enabled = metadata.get("soft_prefix_enabled", False)
        if not isinstance(prefix_enabled, bool):
            raise TypeError("agent graph checkpoint soft-prefix metadata must be boolean")
        if prefix_enabled and self.soft_prefix_graph is None:
            raise ValueError(
                "agent graph checkpoint requires the optional pretrained backend"
            )
        graph = DifferentiableGatedGraph.load(target)
        if (
            graph.node_count != len(self.nodes)
            or graph.latent_size != self.nodes[0].latent_size
        ):
            raise ValueError("agent graph checkpoint dimensions do not match the bridge")
        policy = ClosedLoopRLPolicyOptimizer(
            graph,
            discount=metadata["discount"],
            baseline_rate=metadata["baseline_rate"],
            learning_rate=metadata["learning_rate"],
            max_episode_steps=metadata["max_episode_steps"],
        )
        baseline = metadata["baseline"]
        episode_count = metadata["episode_count"]
        if (
            isinstance(baseline, bool)
            or not isinstance(baseline, (int, float))
            or not math.isfinite(baseline)
            or isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
            or episode_count < 0
        ):
            raise ValueError("agent graph policy metadata is invalid")
        policy.baseline = float(baseline)
        policy.episode_count = episode_count
        prefix_path = target.with_suffix(target.suffix + ".prefix.npz")
        if prefix_enabled and not prefix_path.is_file():
            raise ValueError("agent graph checkpoint is missing its soft-prefix parameters")
        if self.soft_prefix_graph is not None and prefix_enabled:
            try:
                with np.load(prefix_path, allow_pickle=False) as data:
                    state = self.soft_prefix_graph.state_dict()
                    if set(data.files) != set(state):
                        raise ValueError("soft-prefix graph checkpoint has invalid fields")
                    tensors = {
                        name: self.agents.backbone._torch.as_tensor(
                            data[name],
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                        for name, parameter in state.items()
                    }
                    self.soft_prefix_graph.load_state_dict(tensors, strict=True)
            except (OSError, KeyError) as exc:
                raise ValueError(f"could not read soft-prefix graph checkpoint: {exc}") from exc
        self.graph = graph
        self.trainer = DifferentiableGraphGateTrainer(graph)
        self.policy = policy
