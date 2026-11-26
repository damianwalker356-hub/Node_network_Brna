"""Explicitly taught conversational examples with transparent retrieval."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConversationExample:
    prompt: str
    response: str


@dataclass
class ConversationMemory:
    examples: list[ConversationExample]

    def __init__(self, examples=()):
        self.examples = list(examples)

    def teach(self, prompt: str, response: str) -> None:
        prompt = prompt.strip()
        response = response.strip()
        if not prompt or not response:
            raise ValueError("both prompt and response are required")
        example = ConversationExample(prompt, response)
        if example not in self.examples:
            self.examples.append(example)

    def retrieve(self, prompt: str, *, min_similarity: float = 0.25) -> tuple[str, float] | None:
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        query = _tokens(prompt)
        if not query:
            return None
        best_response = None
        best_score = 0.0
        for example in self.examples:
            candidate = _tokens(example.prompt)
            union = query | candidate
            score = len(query & candidate) / len(union) if union else 0.0
            if score > best_score:
                best_response, best_score = example.response, score
        if best_response is None or best_score < min_similarity:
            return None
        return best_response, best_score

    def to_dict(self) -> dict:
        return {"examples": [asdict(example) for example in self.examples]}

    @classmethod
    def from_dict(cls, data: dict) -> ConversationMemory:
        if set(data) != {"examples"} or not isinstance(data["examples"], list):
            raise ValueError("conversation memory must contain an examples list")
        examples = []
        for item in data["examples"]:
            if not isinstance(item, dict) or set(item) != {"prompt", "response"}:
                raise ValueError("each conversation example requires prompt and response")
            examples.append(ConversationExample(item["prompt"], item["response"]))
        return cls(examples)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))
