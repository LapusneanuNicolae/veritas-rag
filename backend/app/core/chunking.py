"""Split text into overlapping chunks for embedding.

A deterministic splitter that prefers to break on paragraph/sentence boundaries,
falling back to a hard character window. ~800 chars with ~120 overlap by default
(roughly the 512-1024 token sweet spot once a reranker is in play)."""
from __future__ import annotations

import re

from app.config import get_settings
from app.types import Chunk

_PARA = re.compile(r"\n\s*\n")


def _split_oversized(block: str, size: int) -> list[str]:
    return [block[i:i + size] for i in range(0, len(block), size)]


def chunk_text(text: str, document: str = "doc", chunk_size: int | None = None,
               overlap: int | None = None) -> list[Chunk]:
    s = get_settings()
    size = chunk_size or s.chunk_size
    ov = overlap if overlap is not None else s.chunk_overlap

    text = text.strip()
    if not text:
        return []

    # 1) group paragraphs greedily up to `size`, keeping structure where possible
    paragraphs = [p.strip() for p in _PARA.split(text) if p.strip()]
    windows: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(p) > size:
            if buf:
                windows.append(buf)
                buf = ""
            windows.extend(_split_oversized(p, size))
            continue
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}" if buf else p
        else:
            windows.append(buf)
            buf = p
    if buf:
        windows.append(buf)

    # 2) add character overlap between consecutive windows for context continuity
    chunks: list[Chunk] = []
    for idx, w in enumerate(windows):
        if idx > 0 and ov > 0:
            tail = windows[idx - 1][-ov:]
            w = tail + w
        chunks.append(Chunk(text=w, index=idx, document=document,
                            location=f"chunk {idx + 1}"))
    return chunks
