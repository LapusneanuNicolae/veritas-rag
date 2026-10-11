# Architecture & design decisions

This document explains *why* Veritas RAG is built the way it is. The goal is a RAG system that is
**accurate, verifiable, and measurable** — not just a demo.

## 1. Retrieval: why hybrid + reranking (not naive cosine)

Pure vector similarity misses exact-match terms (names, codes, acronyms) and is sensitive to how a
question is phrased. Veritas uses:

1. **Hybrid retrieval** — combine semantic (pgvector) and keyword (Postgres full-text) results, then
   fuse scores (Reciprocal Rank Fusion). This catches both "meaning" and "exact term" matches.
2. **Cross-encoder reranking** — re-score the top candidates with a reranker so the chunks sent to
   the LLM are the most relevant few, which directly improves answer quality and lowers token cost.
3. **Query transformation** — rewrite vague questions and decompose complex ones before retrieval.

## 2. Why pgvector (one database)

Instead of running a separate vector DB, embeddings live in PostgreSQL via `pgvector`. One database
means simpler ops, transactional consistency between documents/chunks/metadata, and it keeps the
hybrid (vector + full-text) search in a single place. Good enough well past hobby scale.

## 3. Grounding & anti-hallucination

- The prompt instructs the model to answer **only** from the retrieved context and to return
  "I don't know" otherwise.
- Every answer carries **citations** (document + location) so a human can verify it.
- A verification step (Phase 4) checks the answer against its sources before returning it.

## 4. Model-agnostic by design

LLM and embedding providers sit behind a small interface (`core/llm.py`, `core/embeddings.py`), so
switching between OpenAI, Gemini and Anthropic is a config change, not a rewrite. This avoids vendor
lock-in and makes cost/quality A/B testing possible.

## 5. Evaluation first

A non-deterministic system can't be improved by vibes. A **golden set** of questions with reference
answers is run through RAGAS-style metrics (faithfulness, answer relevancy, context precision/recall).
Any change to chunking, retrieval or prompts is judged against the same set.

## 6. Observability

Each query is traced: which chunks were retrieved, the final prompt, the model used, latency, and
**token cost in USD**. In production, cost and latency matter as much as accuracy.

## 7. Production concerns

JWT auth + multi-tenancy (users only see their own documents), Redis caching of answers, rate
limiting, Dockerized services, CI (lint + tests) on every push, and a live deployment.
