"""RAG service: retrieve relevant chunks, then generate a grounded, cited answer.

Phase 1: dense (vector) retrieval only.
Phase 2 extends `retrieve()` with hybrid search + reranking (see app/core/retrieval.py).
"""
from __future__ import annotations

import time
from typing import Iterator

from app.config import get_settings
from app.core.fusion import reciprocal_rank_fusion
from app.core.query_transform import rewrite_query
from app.core.rerank import get_reranker
from app.observability import trace as obs
from app.providers.factory import get_chat_provider, get_embedding_provider
from app.types import RetrievedChunk
from app.vectorstore.factory import get_vectorstore

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the context "
    "provided. If the answer is not contained in the context, say you don't know. "
    "Cite the sources you used with their [n] markers."
)


def retrieve(question: str, top_n: int | None = None,
             tenant_id: str = "default") -> list[RetrievedChunk]:
    """Full retrieval pipeline: (rewrite) → hybrid search → RRF → rerank → top_n."""
    s = get_settings()
    n = top_n or s.top_n
    store = get_vectorstore()

    search_query = rewrite_query(question) if s.use_query_rewrite else question
    q_vec = get_embedding_provider().embed([search_query])[0]

    dense = store.vector_search(q_vec, top_k=s.top_k, tenant_id=tenant_id)

    if s.use_hybrid:
        lexical = store.keyword_search(search_query, top_k=s.top_k, tenant_id=tenant_id)
        candidates = reciprocal_rank_fusion([dense, lexical])
    else:
        candidates = dense

    if s.use_rerank and candidates:
        return get_reranker().rerank(search_query, candidates, top_n=n)
    return candidates[:n]


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.document} @ {c.location})\n{c.text}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer_with_context(question: str, top_n: int | None = None,
                        tenant_id: str = "default") -> tuple[str, list[RetrievedChunk]]:
    """Return (answer_text, retrieved_chunks) and record a trace for observability."""
    s = get_settings()
    chunks = retrieve(question, top_n=top_n, tenant_id=tenant_id)
    if not chunks:
        return ("I don't have any indexed context yet, so I can't answer reliably. "
                "Upload documents first.", [])

    prompt = build_prompt(question, chunks)
    started = time.perf_counter()
    text = get_chat_provider().complete(SYSTEM_PROMPT, prompt)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    obs.record(obs.QueryTrace(
        question=question,
        rewritten_query=question,
        provider=s.llm_provider,
        model=s.llm_model if s.llm_provider != "fake" else "fake",
        n_chunks=len(chunks),
        documents=sorted({c.document for c in chunks}),
        latency_ms=latency_ms,
        prompt_tokens=obs.estimate_tokens(prompt),
        completion_tokens=obs.estimate_tokens(text),
        cost_usd=obs.estimate_cost(
            s.llm_model if s.llm_provider != "fake" else "fake", prompt, text),
    ))
    return text, chunks


def answer(question: str, top_n: int | None = None,
           tenant_id: str = "default") -> tuple[str, list[dict]]:
    text, chunks = answer_with_context(question, top_n=top_n, tenant_id=tenant_id)
    return text, [c.as_citation() for c in chunks]


def answer_stream(question: str, top_n: int | None = None,
                  tenant_id: str = "default") -> Iterator[str]:
    """Yield the answer token-by-token (used by the SSE endpoint)."""
    chunks = retrieve(question, top_n=top_n, tenant_id=tenant_id)
    if not chunks:
        yield "I don't have any indexed context yet. Upload documents first."
        return
    prompt = build_prompt(question, chunks)
    yield from get_chat_provider().stream(SYSTEM_PROMPT, prompt)
