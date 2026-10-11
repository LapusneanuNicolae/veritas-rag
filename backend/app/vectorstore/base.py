"""VectorStore interface. Two backends implement it: in-memory (dev/tests) and
pgvector (production). Selecting one is a config change."""
from __future__ import annotations

from typing import Protocol

from app.types import RetrievedChunk, StoredChunk


class VectorStore(Protocol):
    def add(self, chunks: list[StoredChunk]) -> None:
        ...

    def vector_search(self, embedding: list[float], top_k: int,
                      tenant_id: str = "default") -> list[RetrievedChunk]:
        ...

    def keyword_search(self, query: str, top_k: int,
                       tenant_id: str = "default") -> list[RetrievedChunk]:
        ...

    def count(self, tenant_id: str | None = None) -> int:
        ...

    def clear(self) -> None:
        ...
