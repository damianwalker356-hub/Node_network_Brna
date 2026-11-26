"""Software-only closed-loop B-RNA simulation.

The reservoir in this module is a seeded numerical model. It does not model
electrical units or drive a physical microelectrode array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brna.network import Critic
from brna.proto import DX, Adam, N, NodeNet, sigmoid

DEFAULT_NODE_ROLES = (
    "sensory",
    "researcher",
    "builder",
    "synthesizer",
    "critic",
    "router",
) * 2


def _validate_batch(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (N, DX):
        raise ValueError(f"{name} must have shape (batch, {N}, {DX})")
    if array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


@dataclass
class EncoderCache:
    inputs: np.ndarray
    node_mixed: np.ndarray
    base: np.ndarray
    profile: np.ndarray


class DigitalEncoder:
    """Maps digital states to abstract temporal stimulus features."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        time_bins: int = 4,
        code: str = "latency",
        adaptive: bool = True,
    ):
        if time_bins < 1:
            raise ValueError("time_bins must be positive")
        if code not in {"latency", "amplitude"}:
            raise ValueError("code must be 'latency' or 'amplitude'")
        self.time_bins = time_bins
        self.code = code
        self.adaptive = adaptive
        self.node_weights = np.eye(N) + rng.normal(0.0, 0.02, (N, N))
        self.feature_weights = np.eye(DX) + rng.normal(0.0, 0.02, (DX, DX))
        self.profile_logits = np.zeros((time_bins, DX))
        if code == "latency":
            for feature in range(DX):
                self.profile_logits[:, feature] = -3.0
                self.profile_logits[feature % time_bins, feature] = 3.0

    @property
    def profile(self) -> np.ndarray:
        shifted = self.profile_logits - np.max(self.profile_logits, axis=0)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=0, keepdims=True)

    @property
    def parameter_count(self) -> int:
        return (
            self.node_weights.size
            + self.feature_weights.size
            + self.profile_logits.size
        )

    def encode(self, digital_state: np.ndarray) -> tuple[np.ndarray, EncoderCache]:
        values = _validate_batch(digital_state, "digital_state")
        node_mixed = np.einsum("ij,bjd->bid", self.node_weights, values)
        base = np.tanh(np.einsum("bid,de->bie", node_mixed, self.feature_weights))
        profile = self.profile
        stimuli = base[None, ...] * profile[:, None, None, :]
        return stimuli, EncoderCache(values, node_mixed, base, profile)

    def backward(
        self,
        cache: EncoderCache,
        stimulus_gradient: np.ndarray,
        *,
        learning_rate: float,
    ) -> np.ndarray:
        d_stimuli = np.asarray(stimulus_gradient, dtype=float)
        if d_stimuli.shape != (
            self.time_bins,
            len(cache.inputs),
            N,
            DX,
        ):
            raise ValueError("stimulus gradient has an incompatible shape")
        base_gradient = np.einsum("tbnd,td->bnd", d_stimuli, cache.profile)
        projected_gradient = base_gradient * (1.0 - cache.base**2)
        node_mixed_gradient = projected_gradient @ self.feature_weights.T
        input_gradient = np.einsum(
            "ij,bid->bjd", self.node_weights, node_mixed_gradient
        )
        if self.adaptive:
            profile_gradient = np.einsum(
                "tbnd,bnd->td", d_stimuli, cache.base
            )
            logits_gradient = cache.profile * (
                profile_gradient
                - np.sum(cache.profile * profile_gradient, axis=0, keepdims=True)
            )
            feature_gradient = np.einsum(
                "bid,bie->de", cache.node_mixed, projected_gradient
            )
            node_gradient = np.einsum(
                "bid,bjd->ij", node_mixed_gradient, cache.inputs
            )
            self.feature_weights -= learning_rate * feature_gradient
            self.node_weights -= learning_rate * node_gradient
            self.profile_logits -= learning_rate * logits_gradient
        return input_gradient


@dataclass
class DecoderCache:
    states: np.ndarray
    pooled: np.ndarray
    node_mixed: np.ndarray


class DigitalDecoder:
    """Maps simulated population activity into node-network features."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        time_bins: int = 4,
        adaptive: bool = True,
    ):
        self.adaptive = adaptive
        self.time_weights = np.zeros(time_bins)
        self.node_weights = np.eye(N) + rng.normal(0.0, 0.02, (N, N))
        self.feature_weights = np.eye(DX) + rng.normal(0.0, 0.02, (DX, DX))

    @property
    def parameter_count(self) -> int:
        return (
            self.time_weights.size
            + self.node_weights.size
            + self.feature_weights.size
        )

    def decode(self, states: np.ndarray) -> tuple[np.ndarray, DecoderCache]:
        activity = np.asarray(states, dtype=float)
        if (
            activity.ndim != 4
            or activity.shape[1] == 0
            or activity.shape[2:] != (N, DX)
            or activity.shape[0] != self.time_weights.size
        ):
            raise ValueError("activity must have shape (time_bins, batch, nodes, features)")
        weights = np.exp(self.time_weights - np.max(self.time_weights))
        weights /= weights.sum()
        pooled = np.einsum("t,tbnd->bnd", weights, activity)
        node_mixed = np.einsum("ij,bjd->bid", self.node_weights, pooled)
        decoded = np.einsum("bid,de->bie", node_mixed, self.feature_weights)
        return decoded, DecoderCache(activity, pooled, node_mixed)

    def backward(
        self,
        cache: DecoderCache,
        decoded_gradient: np.ndarray,
        *,
        learning_rate: float,
    ) -> np.ndarray:
        gradient = np.asarray(decoded_gradient, dtype=float)
        if gradient.shape != cache.pooled.shape:
            raise ValueError("decoded gradient has an incompatible shape")
        node_mixed_gradient = np.einsum(
            "bie,de->bid", gradient, self.feature_weights
        )
        pooled_gradient = np.einsum(
            "ij,bid->bjd", self.node_weights, node_mixed_gradient
        )
        weights = np.exp(self.time_weights - np.max(self.time_weights))
        weights /= weights.sum()
        state_gradient = weights[:, None, None, None] * pooled_gradient[None, ...]
        if self.adaptive:
            feature_gradient = np.einsum(
                "bid,bie->de", cache.node_mixed, gradient
            )
            node_gradient = np.einsum(
                "bid,bjd->ij", node_mixed_gradient, cache.pooled
            )
            weights = np.exp(self.time_weights - np.max(self.time_weights))
            weights /= weights.sum()
            time_gradient = np.einsum(
                "bnd,tbnd->t", pooled_gradient, cache.states
            )
            logits_gradient = weights * (
                time_gradient - np.dot(weights, time_gradient)
            )
            self.feature_weights -= learning_rate * feature_gradient
            self.node_weights -= learning_rate * node_gradient
            self.time_weights -= learning_rate * logits_gradient
        return state_gradient


@dataclass
class ReservoirCache:
    stimuli: np.ndarray
    states: np.ndarray


class SimulatedReservoir:
    """A fixed, noisy recurrent feature generator for software experiments."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        noise_std: float = 0.02,
    ):
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")
        self.rng = rng
        self.noise_std = float(noise_std)
        self.node_weights = rng.normal(0.0, 0.45 / np.sqrt(N), (N, N))
        np.fill_diagonal(self.node_weights, 0.0)
        row_norms = np.sum(np.abs(self.node_weights), axis=1)
        self.node_weights *= min(1.0, 0.8 / max(float(row_norms.max()), 1e-12))
        self.recurrent_features = rng.normal(
            0.0, 0.35 / np.sqrt(DX), (DX, DX)
        )
        self.input_features = np.eye(DX) + rng.normal(
            0.0, 0.03, (DX, DX)
        )

    @property
    def parameter_count(self) -> int:
        return (
            self.node_weights.size
            + self.recurrent_features.size
            + self.input_features.size
        )

    def respond(self, stimuli: np.ndarray) -> tuple[np.ndarray, ReservoirCache]:
        codes = np.asarray(stimuli, dtype=float)
        if (
            codes.ndim != 4
            or codes.shape[0] == 0
            or codes.shape[1] == 0
            or codes.shape[2:] != (N, DX)
            or not np.isfinite(codes).all()
        ):
            raise ValueError("stimuli must have shape (time_bins, batch, nodes, features)")
        previous = np.zeros_like(codes[0])
        states = []
        for stimulus in codes:
            recurrent = np.einsum(
                "ij,bjd,de->bie",
                self.node_weights,
                previous,
                self.recurrent_features,
            )
            noise = self.rng.normal(0.0, self.noise_std, stimulus.shape)
            previous = np.tanh(
                stimulus @ self.input_features + recurrent + noise
            )
            states.append(previous)
        activity = np.stack(states)
        return activity, ReservoirCache(codes, activity)

    def backward(
        self,
        cache: ReservoirCache,
        activity_gradient: np.ndarray,
    ) -> np.ndarray:
        gradient = np.asarray(activity_gradient, dtype=float)
        if gradient.shape != cache.states.shape:
            raise ValueError("activity gradient has an incompatible shape")
        stimulus_gradient = np.zeros_like(cache.stimuli)
        previous_gradient = np.zeros_like(cache.states[0])
        for time_index in range(len(cache.states) - 1, -1, -1):
            state = cache.states[time_index]
            state_gradient = activity_gradient[time_index] + previous_gradient
            preactivation_gradient = state_gradient * (1.0 - state**2)
            stimulus_gradient[time_index] = (
                preactivation_gradient @ self.input_features.T
            )
            previous_gradient = np.einsum(
                "ij,bie,de->bjd",
                self.node_weights,
                preactivation_gradient,
                self.recurrent_features,
            )
        return stimulus_gradient


class ClosedLoopBRNA:
    """Role-node network coupled to a simulated reservoir and digital translators."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        code: str = "latency",
        adaptive_translators: bool = True,
        outer_loops: int = 2,
        noise_std: float = 0.02,
        roles: tuple[str, ...] = DEFAULT_NODE_ROLES,
    ):
        if isinstance(outer_loops, bool) or not 1 <= outer_loops <= 4:
            raise ValueError("outer_loops must be from 1 to 4")
        self.outer_loops = outer_loops
        self.encoder = DigitalEncoder(
            rng, code=code, adaptive=adaptive_translators
        )
        self.reservoir = SimulatedReservoir(rng, noise_std=noise_std)
        self.decoder = DigitalDecoder(rng, adaptive=adaptive_translators)
        self.network = NodeNet(rng, roles=roles)
        self.node_roles = self.network.roles

    @property
    def parameter_count(self) -> int:
        translator_parameters = (
            self.encoder.parameter_count + self.decoder.parameter_count
            if self.encoder.adaptive
            else 0
        )
        return self.network.nparams() + translator_parameters

    @property
    def system_parameter_count(self) -> int:
        return (
            self.network.nparams()
            + self.encoder.parameter_count
            + self.decoder.parameter_count
            + self.reservoir.parameter_count
        )

    def _one_round(
        self, digital_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, EncoderCache, ReservoirCache, DecoderCache]:
        stimuli, encoder_cache = self.encoder.encode(digital_state)
        activity, reservoir_cache = self.reservoir.respond(stimuli)
        decoded, decoder_cache = self.decoder.decode(activity)
        states, _, _, _, logits = self.network.fwd(decoded)
        return logits, states[-1], encoder_cache, reservoir_cache, decoder_cache

    def run(self, digital_state: np.ndarray) -> dict[str, object]:
        original = _validate_batch(digital_state, "digital_state")
        current = original.copy()
        critic = Critic(
            hop_limit=self.outer_loops,
            budget=max(0, self.outer_loops - 1),
        )
        logits = np.zeros(len(original))
        for hop in range(self.outer_loops):
            logits, node_state, _, _, _ = self._one_round(current)
            if hop + 1 < self.outer_loops and critic.allow(hop, hop + 1, hop):
                feedback = node_state[:, :, :DX]
                current = np.tanh(original + feedback)
        return {
            "logits": logits,
            "feedback_edges": critic.count,
            "rounds": critic.count + 1,
            "node_roles": self.node_roles,
        }

    def train_step(
        self,
        digital_state: np.ndarray,
        labels: np.ndarray,
        *,
        translator_learning_rate: float = 0.001,
    ) -> float:
        original = _validate_batch(digital_state, "digital_state")
        targets = np.asarray(labels, dtype=float)
        if (
            targets.shape != (len(original),)
            or not np.isfinite(targets).all()
            or not np.isin(targets, (0.0, 1.0)).all()
        ):
            raise ValueError("labels must be a finite binary value per input")
        if not np.isfinite(translator_learning_rate) or translator_learning_rate < 0:
            raise ValueError("translator_learning_rate must be finite and non-negative")

        current = original.copy()
        critic = Critic(
            hop_limit=self.outer_loops,
            budget=max(0, self.outer_loops - 1),
        )
        final_loss = 0.0
        for hop in range(self.outer_loops):
            stimuli, encoder_cache = self.encoder.encode(current)
            activity, reservoir_cache = self.reservoir.respond(stimuli)
            decoded, decoder_cache = self.decoder.decode(activity)
            network_states, _, _, _, logits = self.network.fwd(decoded)
            if hop == self.outer_loops - 1:
                final_loss = float(
                    np.mean(np.logaddexp(0.0, logits) - targets * logits)
                )
                decoded_gradient = self.network.train_step(
                    decoded, targets, return_input_gradient=True
                )
                if decoded_gradient is None:
                    raise RuntimeError(
                        "node network did not return its input gradient"
                    )
                activity_gradient = self.decoder.backward(
                    decoder_cache,
                    decoded_gradient,
                    learning_rate=translator_learning_rate,
                )
                stimulus_gradient = self.reservoir.backward(
                    reservoir_cache, activity_gradient
                )
                self.encoder.backward(
                    encoder_cache,
                    stimulus_gradient,
                    learning_rate=translator_learning_rate,
                )
            if hop + 1 < self.outer_loops and critic.allow(hop, hop + 1, hop):
                current = np.tanh(original + network_states[-1][:, :, :DX])
        return final_loss


class BudgetMLP:
    """A silicon-only classifier with an exact requested parameter budget."""

    def __init__(self, parameter_budget: int, rng: np.random.Generator):
        input_size = N * DX
        hidden_size = (parameter_budget - 1) // (input_size + 2)
        if hidden_size < 1:
            raise ValueError("parameter_budget is too small")
        base_parameters = (input_size + 2) * hidden_size + 1
        residual_count = parameter_budget - base_parameters
        self.hidden_weights = rng.normal(
            0.0, 1.0 / np.sqrt(input_size), (input_size, hidden_size)
        )
        self.hidden_bias = np.zeros(hidden_size)
        self.output_weights = rng.normal(
            0.0, 1.0 / np.sqrt(hidden_size), hidden_size
        )
        self.output_bias = np.zeros(1)
        self.residual_indices = np.arange(residual_count) % input_size
        self.residual_weights = np.zeros(residual_count)
        self.parameters = [
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
            self.residual_weights,
        ]
        self.optimizer = Adam(self.parameters)
        self.parameter_budget = parameter_budget

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self.parameters)

    def logits(self, values: np.ndarray) -> np.ndarray:
        inputs = _validate_batch(values, "digital_state").reshape(len(values), -1)
        hidden = np.tanh(inputs @ self.hidden_weights + self.hidden_bias)
        output = hidden @ self.output_weights + self.output_bias
        if self.residual_weights.size:
            output += inputs[:, self.residual_indices] @ self.residual_weights
        return output

    def train_step(self, values: np.ndarray, labels: np.ndarray) -> None:
        inputs = _validate_batch(values, "digital_state").reshape(len(values), -1)
        targets = np.asarray(labels, dtype=float)
        if targets.shape != (len(inputs),) or not np.isin(targets, (0.0, 1.0)).all():
            raise ValueError("labels must be binary values matching the input batch")
        hidden = np.tanh(inputs @ self.hidden_weights + self.hidden_bias)
        output = hidden @ self.output_weights + self.output_bias
        if self.residual_weights.size:
            output += inputs[:, self.residual_indices] @ self.residual_weights
        output_gradient = (sigmoid(output) - targets) / len(inputs)
        hidden_gradient = (
            output_gradient[:, None]
            * self.output_weights
            * (1.0 - hidden**2)
        )
        residual_gradient = (
            inputs[:, self.residual_indices].T @ output_gradient
            if self.residual_weights.size
            else np.zeros_like(self.residual_weights)
        )
        self.optimizer.step(
            [
                inputs.T @ hidden_gradient,
                hidden_gradient.sum(axis=0),
                hidden.T @ output_gradient,
                np.array([output_gradient.sum()]),
                residual_gradient,
            ]
        )


class RandomReservoirReadout:
    """A trained linear readout over fixed simulated-reservoir features."""

    def __init__(self, rng: np.random.Generator, *, code: str = "latency"):
        self.encoder = DigitalEncoder(rng, code=code, adaptive=False)
        self.reservoir = SimulatedReservoir(rng)
        self.decoder = DigitalDecoder(rng, adaptive=False)
        self.weights = rng.normal(0.0, 0.02, N * DX)
        self.bias = np.zeros(1)
        self.learning_rate = 0.05

    @property
    def parameter_count(self) -> int:
        return self.weights.size + self.bias.size

    @property
    def system_parameter_count(self) -> int:
        return (
            self.encoder.parameter_count
            + self.reservoir.parameter_count
            + self.decoder.parameter_count
            + self.parameter_count
        )

    def logits(self, values: np.ndarray) -> np.ndarray:
        stimuli, _ = self.encoder.encode(values)
        activity, _ = self.reservoir.respond(stimuli)
        decoded, _ = self.decoder.decode(activity)
        return decoded.reshape(len(values), -1) @ self.weights + self.bias

    def train_step(self, values: np.ndarray, labels: np.ndarray) -> None:
        inputs = _validate_batch(values, "digital_state")
        targets = np.asarray(labels, dtype=float)
        if targets.shape != (len(inputs),) or not np.isin(targets, (0.0, 1.0)).all():
            raise ValueError("labels must be binary values matching the input batch")
        stimuli, _ = self.encoder.encode(inputs)
        activity, _ = self.reservoir.respond(stimuli)
        decoded, _ = self.decoder.decode(activity)
        features = decoded.reshape(len(inputs), -1)
        output = features @ self.weights + self.bias
        output_gradient = (sigmoid(output) - targets) / len(inputs)
        self.weights -= self.learning_rate * (features.T @ output_gradient)
        self.bias -= self.learning_rate * np.array([output_gradient.sum()])
