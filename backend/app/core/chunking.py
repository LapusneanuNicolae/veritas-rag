"""Split text into overlapping chunks for embedding.

A simple, deterministic character splitter with overlap. Phase 2 will add
structure-aware chunking (headings, paragraphs) for better retrieval.
"""
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class Chunk:
    text: str
    index: int


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start, idx = 0, 0
    while start < len(text):
        end = start + size
        chunks.append(Chunk(text=text[start:end], index=idx))
        idx += 1
        start += size - ov
    return chunks
