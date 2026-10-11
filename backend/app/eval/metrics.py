"""RAG evaluation metrics (RAGAS-style).

Four metrics, each targeting a distinct failure mode:
  - faithfulness       : is the answer grounded in the retrieved context? (hallucination)
  - answer_relevancy   : does the answer address the question?
  - context_precision  : are the retrieved chunks relevant? (retrieval noise)
  - context_recall     : does the context cover the reference answer? (retrieval gaps)

Production RAGAS uses an LLM judge. To keep the framework runnable & deterministic in
CI, this implements embedding- and lexical-based approximations behind the same API.
The scoring backend can be swapped for an LLM judge without changing callers.
"""
from __future__ import annotations

import re

from app.providers.factory import get_embedding_provider

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "it", "this", "that", "with", "as", "at", "by", "be", "can",
    "what", "who", "when", "where", "which", "how", "does", "do", "i", "you",
}


def _content(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """How well the answer is grounded in the context.

    Scored as the mean, over answer sentences, of the fraction of each sentence's
    content words that appear in the retrieved context. A fully grounded answer scores
    ~1.0; a sentence introducing unsupported facts drags the score down.
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0
    ctx = _content(" ".join(contexts))
    if not ctx:
        return 0.0
    ratios: list[float] = []
    for s in sents:
        words = _content(s)
        if not words:
            ratios.append(1.0)
        else:
            ratios.append(len(words & ctx) / len(words))
    return sum(ratios) / len(ratios)


def answer_relevancy(question: str, answer: str) -> float:
    """Semantic similarity between the question and the answer (via embeddings)."""
    if not answer.strip():
        return 0.0
    emb = get_embedding_provider().embed([question, answer])
    return max(0.0, min(1.0, _cosine(emb[0], emb[1])))


def context_precision(contexts: list[str], reference: str) -> float:
    """Fraction of retrieved chunks that overlap the reference answer."""
    if not contexts:
        return 0.0
    ref = _content(reference)
    if not ref:
        return 0.0
    relevant = sum(1 for c in contexts if _content(c) & ref)
    return relevant / len(contexts)


def context_recall(contexts: list[str], reference: str) -> float:
    """Fraction of the reference answer's content covered by the context."""
    ref = _content(reference)
    if not ref:
        return 1.0
    ctx = _content(" ".join(contexts))
    return len(ref & ctx) / len(ref)
