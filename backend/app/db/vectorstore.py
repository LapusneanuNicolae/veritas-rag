"""pgvector-backed storage for chunks + embeddings (Phase 1).

Schema (created by migrations/init):
  documents(id, filename, tenant_id, created_at)
  chunks(id, document_id, idx, content, location, embedding vector(N),
         content_tsv tsvector)   -- tsvector enables hybrid keyword search

This module will expose: store_chunks(), vector_search(), keyword_search().
"""
from app.config import get_settings


def store_chunks(document_id: str, chunks: list, embeddings: list[list[float]]) -> None:
    """Persist chunks and their embeddings. TODO(Phase 1)."""
    _ = (document_id, chunks, embeddings, get_settings())
    raise NotImplementedError


def vector_search(embedding: list[float], top_k: int) -> list[dict]:
    """Cosine-nearest chunks via pgvector. TODO(Phase 1)."""
    _ = (embedding, top_k)
    raise NotImplementedError


def keyword_search(query: str, top_k: int) -> list[dict]:
    """Full-text keyword search for hybrid retrieval. TODO(Phase 2)."""
    _ = (query, top_k)
    raise NotImplementedError
