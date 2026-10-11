"""Retrieval pipeline: hybrid search + reranking.

Target design (phased):
  1. query transformation (rewrite / decompose)       [Phase 2]
  2. vector search (pgvector) + keyword search (FTS)   [Phase 1 / 2]
  3. fuse results (Reciprocal Rank Fusion)             [Phase 2]
  4. cross-encoder rerank -> top_n                     [Phase 2]

Returns the chunks to ground the answer on, each with source metadata for citations.
"""
from app.config import get_settings


def retrieve(question: str, top_n: int | None = None) -> list[dict]:
    """Return the most relevant chunks for a question.

    Each chunk: {"text", "document", "location", "score"}.
    """
    settings = get_settings()
    n = top_n or settings.top_n

    # TODO(Phase 1): embed question, vector-search pgvector for top_k candidates
    # TODO(Phase 2): add keyword search, RRF fusion, and cross-encoder reranking
    _ = (question, n)
    return []


def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse multiple ranked lists of chunk ids into a single score map (Phase 2)."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, chunk_id in enumerate(results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
