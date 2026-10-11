"""Select and cache the configured vector store (singleton)."""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.vectorstore.base import VectorStore


@lru_cache
def get_vectorstore() -> VectorStore:
    s = get_settings()
    if s.vector_backend == "memory":
        from app.vectorstore.memory import InMemoryVectorStore
        return InMemoryVectorStore()
    if s.vector_backend == "pgvector":
        from app.vectorstore.pgvector import PgVectorStore
        return PgVectorStore()
    raise ValueError(f"Unknown vector_backend: {s.vector_backend}")
