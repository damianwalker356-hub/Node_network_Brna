"""A small, trainable character-level recurrent language model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

BOS = 0
EOS = 1
UNK = 2
SPECIAL_TOKENS = 3
MAX_VOCABULARY_SIZE = 2_048
MIN_HIDDEN_SIZE = 8
MAX_HIDDEN_SIZE = 128
MAX_TRAINING_CHARACTERS = 50_000
MAX_SEQUENCE_LENGTH = 256


class CharacterLanguageModel:
    """Train a compact Elman RNN from scratch on Unicode text."""

    def __init__(self, *, hidden_size: int = 32, seed: int = 1) -> None:
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or not MIN_HIDDEN_SIZE <= hidden_size <= MAX_HIDDEN_SIZE
        ):
            raise ValueError(f"hidden_size must be from {MIN_HIDDEN_SIZE} to {MAX_HIDDEN_SIZE}")
        self.hidden_size = hidden_size
        self.trained_characters = 0
        self.vocabulary: list[str] = []
        self._rng = np.random.default_rng(seed)
        self.input_weights = np.zeros((SPECIAL_TOKENS, hidden_size))
        self.recurrent_weights = self._rng.normal(
            0, np.sqrt(1 / hidden_size), (hidden_size, hidden_size)
        )
        self.hidden_bias = np.zeros(hidden_size)
        self.output_weights = np.zeros((hidden_size, SPECIAL_TOKENS))
        self.output_bias = np.zeros(SPECIAL_TOKENS)

    def train(
        self,
        text: str,
        *,
        epochs: int = 3,
        learning_rate: float = 0.05,
        sequence_length: int = 96,
    ) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("training text must be a string")
        if len(text) > MAX_TRAINING_CHARACTERS:
            raise ValueError(
                f"training text must not exceed {MAX_TRAINING_CHARACTERS} characters"
            )
        if len(text) < 2:
            raise ValueError("training text must contain at least two characters")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be a whole number from 1 to 50")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not np.isfinite(learning_rate)
            or not 0 < learning_rate <= 1
        ):
            raise ValueError("learning_rate must be finite and between 0 and 1")
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or not 8 <= sequence_length <= MAX_SEQUENCE_LENGTH
        ):
            raise ValueError(f"sequence_length must be from 8 to {MAX_SEQUENCE_LENGTH}")

        self._extend_vocabulary(text)
        encoded = self._encode(text)
        sequences = [
            encoded[index : index + sequence_length]
            for index in range(0, len(encoded), sequence_length)
        ]
        losses = []
        for _ in range(epochs):
            order = self._rng.permutation(len(sequences))
            total_loss = 0.0
            token_count = 0
            for sequence_index in order:
                loss, gradients, count = self._sequence_gradients(sequences[sequence_index])
                norm = np.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
                scale = min(1.0, 5.0 / max(norm, 1e-12))
                for parameter, gradient in zip(self._parameters(), gradients, strict=True):
                    parameter -= learning_rate * scale * gradient
                total_loss += loss
                token_count += count
            losses.append(total_loss / token_count)
        self.trained_characters += len(text) * epochs
        return losses

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 240,
        temperature: float = 0.8,
        top_k: int = 32,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.trained_characters == 0:
            raise RuntimeError("model has not been trained yet")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= 1024
        ):
            raise ValueError("max_new_tokens must be a whole number from 1 to 1024")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature)
            or not 0.1 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0.1 and 2")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 128:
            raise ValueError("top_k must be a whole number from 1 to 128")

        hidden = np.zeros(self.hidden_size)
        token = BOS
        for value in self._encode(prompt[-MAX_SEQUENCE_LENGTH:]):
            hidden = self._step(token, hidden)
            token = value
        hidden = self._step(token, hidden)
        generated = []
        for _ in range(max_new_tokens):
            logits = hidden @ self.output_weights + self.output_bias
            logits[[BOS, UNK]] = -np.inf
            logits = logits / temperature
            count = min(top_k, len(logits) - 2)
            selected = np.argpartition(logits, -count)[-count:]
            probabilities = _softmax(logits[selected])
            token = int(self._rng.choice(selected, p=probabilities))
            hidden = self._step(token, hidden)
            if token == EOS:
                if len(generated) >= 8:
                    break
                continue
            generated.append(self.vocabulary[token - SPECIAL_TOKENS])
        return "".join(generated).strip()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            np.savez_compressed(
                stream,
                version=np.asarray([2], dtype=np.int64),
                hidden_size=np.asarray([self.hidden_size], dtype=np.int64),
                trained_characters=np.asarray([self.trained_characters], dtype=np.int64),
                vocabulary=np.asarray(self.vocabulary, dtype=np.str_),
                input_weights=self.input_weights,
                recurrent_weights=self.recurrent_weights,
                hidden_bias=self.hidden_bias,
                output_weights=self.output_weights,
                output_bias=self.output_bias,
            )

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 1) -> CharacterLanguageModel:
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data["version"][0]) != 2:
                    raise ValueError("unsupported language-model file version")
                model = cls(hidden_size=int(data["hidden_size"][0]), seed=seed)
                vocabulary = data["vocabulary"].tolist()
                if (
                    not isinstance(vocabulary, list)
                    or len(vocabulary) > MAX_VOCABULARY_SIZE
                    or any(not isinstance(char, str) or len(char) != 1 for char in vocabulary)
                    or len(set(vocabulary)) != len(vocabulary)
                ):
                    raise ValueError("model vocabulary is invalid")
                model.vocabulary = vocabulary
                model.trained_characters = int(data["trained_characters"][0])
                if model.trained_characters < 0 or (model.trained_characters and not vocabulary):
                    raise ValueError("training metadata does not match the model vocabulary")
                model._resize_parameters(len(vocabulary) + SPECIAL_TOKENS)
                expected_shapes = (
                    (model.input_weights, data["input_weights"]),
                    (model.recurrent_weights, data["recurrent_weights"]),
                    (model.hidden_bias, data["hidden_bias"]),
                    (model.output_weights, data["output_weights"]),
                    (model.output_bias, data["output_bias"]),
                )
                for expected, value in expected_shapes:
                    if value.shape != expected.shape or not np.isfinite(value).all():
                        raise ValueError("model parameters have invalid shape or values")
                    expected[...] = value
                return model
        except (OSError, KeyError, IndexError) as exc:
            raise ValueError(f"could not read language-model file: {exc}") from exc

    def _parameters(self) -> Sequence[np.ndarray]:
        return (
            self.input_weights,
            self.recurrent_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
        )

    def _extend_vocabulary(self, text: str) -> None:
        existing_characters = set(self.vocabulary)
        new_characters = []
        for character in text:
            if character not in existing_characters:
                new_characters.append(character)
                existing_characters.add(character)
        if len(self.vocabulary) + len(new_characters) > MAX_VOCABULARY_SIZE:
            raise ValueError(f"model vocabulary cannot exceed {MAX_VOCABULARY_SIZE} characters")
        if not new_characters:
            return
        old_size = len(self.vocabulary) + SPECIAL_TOKENS
        self.vocabulary.extend(new_characters)
        self._resize_parameters(len(self.vocabulary) + SPECIAL_TOKENS, old_size=old_size)

    def _resize_parameters(self, vocab_size: int, *, old_size: int | None = None) -> None:
        input_weights = self._rng.normal(
            0, np.sqrt(1 / self.hidden_size), (vocab_size, self.hidden_size)
        )
        output_weights = self._rng.normal(
            0, np.sqrt(1 / self.hidden_size), (self.hidden_size, vocab_size)
        )
        output_bias = np.zeros(vocab_size)
        if old_size is not None:
            input_weights[:old_size] = self.input_weights
            output_weights[:, :old_size] = self.output_weights
            output_bias[:old_size] = self.output_bias
        self.input_weights = input_weights
        self.output_weights = output_weights
        self.output_bias = output_bias

    def _encode(self, text: str) -> list[int]:
        vocabulary = {char: index + SPECIAL_TOKENS for index, char in enumerate(self.vocabulary)}
        return [vocabulary.get(char, UNK) for char in text]

    def _step(self, token: int, previous: np.ndarray) -> np.ndarray:
        return np.tanh(self.input_weights[token] + previous @ self.recurrent_weights + self.hidden_bias)

    def _sequence_gradients(self, sequence: Sequence[int]) -> tuple[float, tuple[np.ndarray, ...], int]:
        tokens = [BOS, *sequence]
        targets = [*sequence, EOS]
        hidden_states = [np.zeros(self.hidden_size)]
        probabilities = []
        loss = 0.0
        for token, target in zip(tokens, targets, strict=True):
            hidden = self._step(token, hidden_states[-1])
            hidden_states.append(hidden)
            probs = _softmax(hidden @ self.output_weights + self.output_bias)
            loss -= float(np.log(max(probs[target], 1e-12)))
            probabilities.append((probs, target))

        input_gradient = np.zeros_like(self.input_weights)
        recurrent_gradient = np.zeros_like(self.recurrent_weights)
        hidden_bias_gradient = np.zeros_like(self.hidden_bias)
        output_gradient = np.zeros_like(self.output_weights)
        output_bias_gradient = np.zeros_like(self.output_bias)
        next_hidden_gradient = np.zeros(self.hidden_size)
        for index in range(len(targets) - 1, -1, -1):
            probs, target = probabilities[index]
            probs[target] -= 1
            output_gradient += np.outer(hidden_states[index + 1], probs)
            output_bias_gradient += probs
            hidden_gradient = probs @ self.output_weights.T + next_hidden_gradient
            activation_gradient = hidden_gradient * (1 - hidden_states[index + 1] ** 2)
            input_gradient[tokens[index]] += activation_gradient
            recurrent_gradient += np.outer(hidden_states[index], activation_gradient)
            hidden_bias_gradient += activation_gradient
            next_hidden_gradient = activation_gradient @ self.recurrent_weights.T

        count = len(targets)
        gradients = (
            input_gradient / count,
            recurrent_gradient / count,
            hidden_bias_gradient / count,
            output_gradient / count,
            output_bias_gradient / count,
        )
        return loss, gradients, count


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)
