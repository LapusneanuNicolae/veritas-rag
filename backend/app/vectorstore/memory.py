"""In-memory vector store: numpy cosine similarity for dense search and a simple
BM25-style scorer for keyword search. Zero external dependencies — used for local
dev, tests and CI. Not persistent."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from app.types import RetrievedChunk, StoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class InMemoryVectorStore:
    def __init__(self):
        self._chunks: list[StoredChunk] = []

    def add(self, chunks: list[StoredChunk]) -> None:
        self._chunks.extend(chunks)

    def _scope(self, tenant_id: str | None) -> list[StoredChunk]:
        if tenant_id is None:
            return self._chunks
        return [c for c in self._chunks if c.tenant_id == tenant_id]

    def vector_search(self, embedding: list[float], top_k: int,
                      tenant_id: str = "default") -> list[RetrievedChunk]:
        scope = self._scope(tenant_id)
        if not scope:
            return []
        q = np.asarray(embedding, dtype=float)
        qn = np.linalg.norm(q) or 1.0
        scored: list[RetrievedChunk] = []
        for c in scope:
            v = np.asarray(c.embedding, dtype=float)
            denom = (np.linalg.norm(v) or 1.0) * qn
            score = float(np.dot(q, v) / denom)
            scored.append(RetrievedChunk(text=c.text, document=c.document,
                                         location=c.location, score=score,
                                         chunk_id=c.chunk_id))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def keyword_search(self, query: str, top_k: int,
                       tenant_id: str = "default") -> list[RetrievedChunk]:
        scope = self._scope(tenant_id)
        if not scope:
            return []
        q_terms = _tok(query)
        if not q_terms:
            return []
        docs = [_tok(c.text) for c in scope]
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / n
        # document frequency
        df: Counter = Counter()
        for d in set_terms_per_doc(docs):
            for t in d:
                df[t] += 1
        k1, b = 1.5, 0.75
        results: list[RetrievedChunk] = []
        for c, d in zip(scope, docs):
            freq = Counter(d)
            score = 0.0
            for t in q_terms:
                if t not in freq:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                tf = freq[t]
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(d) / avgdl))
            if score > 0:
                results.append(RetrievedChunk(text=c.text, document=c.document,
                                              location=c.location, score=score,
                                              chunk_id=c.chunk_id))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def count(self, tenant_id: str | None = None) -> int:
        return len(self._scope(tenant_id))

    def clear(self) -> None:
        self._chunks.clear()


def set_terms_per_doc(docs: list[list[str]]) -> list[set[str]]:
    return [set(d) for d in docs]
