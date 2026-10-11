"""Deterministic, dependency-free providers for local dev, tests and CI.

The fake embedder is a hashing bag-of-words vectorizer: texts sharing words get
similar vectors, so cosine similarity is actually meaningful in tests — without any
API key or network call. The fake chat model returns a grounded, templated answer.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterator

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _stable_hash(token: str) -> int:
    """Process-independent hash (unlike built-in hash(), which is salted)."""
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big")


class FakeEmbeddingProvider:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _tokens(text):
                vec[_stable_hash(tok) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FakeChatProvider:
    """Returns the most relevant sentence from the context + a citation marker.

    Good enough to exercise the full pipeline deterministically. If the context is
    empty it declines to answer (anti-hallucination behaviour under test)."""

    def complete(self, system: str, user: str) -> str:
        return "".join(self.stream(system, user))

    def stream(self, system: str, user: str) -> Iterator[str]:
        # Transform-style tasks (query rewrite / decomposition) have no Context block:
        # echo the input so those steps behave sensibly without a real model.
        if "Context:" not in user:
            yield user.strip()
            return
        context = ""
        if "Context:" in user and "Question:" in user:
            context = user.split("Context:", 1)[1].split("Question:", 1)[0]
        if not context.strip():
            yield "I don't know based on the provided context."
            return
        # pick the first non-empty context line as the "answer", cite [1]
        first = next((ln.strip() for ln in context.splitlines() if ln.strip()
                      and not ln.strip().startswith("[")), "")
        words = (f"Based on the context: {first} [1]").split(" ")
        for i, w in enumerate(words):
            yield (w if i == 0 else " " + w)
