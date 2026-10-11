"""Rerankers re-score retrieved candidates by true query↔passage relevance and keep
the best few. This is the single biggest lever on RAG quality.

- FakeReranker: deterministic lexical-overlap reranker (no deps) for dev/tests/CI.
- CrossEncoderReranker: a real cross-encoder (sentence-transformers), used in prod.
"""
from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

from app.config import get_settings
from app.types import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class FakeReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]:
        q = _tok(query)
        rescored = []
        for c in chunks:
            overlap = len(q & _tok(c.text))
            # primary: lexical overlap; tiny tie-break from the upstream score
            rescored.append(replace(c, score=overlap + c.score * 1e-3))
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_n]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy / optional dep
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_n]


@lru_cache
def get_reranker():
    s = get_settings()
    if s.rerank_provider == "cross-encoder":
        return CrossEncoderReranker()
    return FakeReranker()
