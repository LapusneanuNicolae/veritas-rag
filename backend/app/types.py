"""Shared domain types used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A piece of a document, ready to embed."""
    text: str
    index: int
    document: str
    location: str  # e.g. "chunk 3" or "page 5"
    embedding: list[float] | None = None


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with a relevance score and source metadata."""
    text: str
    document: str
    location: str
    score: float
    chunk_id: str = ""

    def as_citation(self) -> dict:
        return {
            "document": self.document,
            "location": self.location,
            "snippet": (self.text[:160] + "…") if len(self.text) > 160 else self.text,
        }


@dataclass
class StoredChunk:
    """A chunk persisted in the vector store."""
    chunk_id: str
    document: str
    location: str
    text: str
    embedding: list[float]
    tenant_id: str = "default"
    extra: dict = field(default_factory=dict)
