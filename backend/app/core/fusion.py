"""Reciprocal Rank Fusion (RRF) — combine several ranked lists into one.

RRF is the standard, robust way to merge dense (vector) and lexical (keyword) results
without having to calibrate their very different score scales.
"""
from __future__ import annotations

from app.types import RetrievedChunk


def _key(c: RetrievedChunk) -> str:
    return c.chunk_id or f"{c.document}|{c.location}"


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = 60,
                           top: int | None = None) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, rc in enumerate(results):
            key = _key(rc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(key, rc)

    fused = [
        RetrievedChunk(text=best[key].text, document=best[key].document,
                       location=best[key].location, score=score,
                       chunk_id=best[key].chunk_id)
        for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return fused[:top] if top else fused
