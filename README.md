<h1 align="center">🔎 Veritas RAG</h1>

<p align="center">
  <b>A production-grade, model-agnostic Retrieval-Augmented Generation platform</b><br>
  Answers questions over your documents with <b>verifiable citations</b> — built around
  hybrid retrieval, reranking, an agentic layer, and a real evaluation framework.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20active%20development-orange">
  <img src="https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/vector%20db-pgvector-4169E1?logo=postgresql&logoColor=white">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why this project

Most RAG demos are a thin wrapper around an LLM: `embed → cosine similarity → prompt`.
In practice, the hard parts are **retrieval quality**, **evaluation of a non-deterministic system**,
and **not hallucinating**. Veritas RAG is built to take those seriously:

- **Grounded answers with citations** — every answer links back to the source chunk; the model is
  instructed to say *"I don't know"* when the context doesn't contain the answer.
- **Advanced retrieval** — hybrid search (vector + keyword) with cross-encoder **reranking**, plus
  query rewriting / decomposition.
- **Agentic layer** — a multi-step agent that decides when to retrieve and self-corrects.
- **Evaluation framework** — a golden set + metrics (faithfulness, answer relevancy, context recall)
  so retrieval strategies can be compared, not guessed.
- **Observability** — tracing of retrieved chunks, latency and **token cost per query**.
- **Production concerns** — JWT auth, multi-tenancy, Redis caching, Docker, CI, tests.

> This is a learning-grade project built in the open to demonstrate engineering depth in RAG.

---

## Architecture

```
          ┌──────────────────────────────┐
          │       Frontend (Next.js)      │
          │ chat · upload · sources ·     │
          │ evals / cost dashboard        │
          └───────────────┬──────────────┘
                          │ REST / SSE (streaming)
          ┌───────────────▼──────────────┐
          │       API (FastAPI)           │
          │ auth · rate-limit · tenancy   │
          └───┬──────────────────────┬────┘
              │                      │
   ┌──────────▼─────────┐  ┌─────────▼──────────────┐
   │ Ingestion pipeline  │  │     RAG engine          │
   │ parse·chunk·embed · │  │ rewrite → hybrid        │
   │ async workers       │  │ retrieve → rerank →     │
   └──────────┬──────────┘  │ LLM → verify/cite       │
              │             └───────────┬────────────┘
              ▼                         │
   ┌──────────────────────┐            │
   │ PostgreSQL + pgvector │◄───────────┘
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐   ┌──────────────────────┐
   │   Redis (cache/queue) │   │ Observability (trace, │
   └──────────────────────┘   │ token cost, evals)    │
                              └──────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and trade-offs.

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | Next.js · React · Tailwind CSS |
| Backend | Python · FastAPI |
| LLM orchestration | LangChain / LangGraph |
| Vector DB | PostgreSQL + pgvector |
| Keyword search | Postgres full-text (hybrid) |
| LLM & embeddings | OpenAI / Gemini / Anthropic (**model-agnostic**, via config) |
| Cache / queue | Redis |
| Evaluation | golden set + RAGAS-style metrics |
| Infra | Docker Compose · GitHub Actions CI |
| Tests | pytest |

---

## Roadmap

- [x] **Phase 1 — MVP:** upload (PDF/DOCX/MD/TXT), ingestion pipeline, vector retrieval, cited
  answers, streaming, "I don't know" · *pluggable providers (fake/OpenAI) + vector backends
  (in-memory/pgvector) so it runs & tests with zero external services*
- [ ] **Phase 2 — Advanced retrieval:** hybrid search, cross-encoder reranking, query rewriting / HyDE
- [ ] **Phase 3 — Evaluation & observability:** golden set, metrics, eval dashboard, cost/latency tracing
- [ ] **Phase 4 — Agentic:** LangGraph agent, query decomposition, self-correction, tools
- [ ] **Phase 5 — Production:** JWT auth, multi-tenancy, Redis caching, CI, live deploy

---

## Quick start (local)

> Requires Docker. Copy `.env.example` to `.env` and add your LLM API key.

```bash
cp .env.example .env         # then edit .env with your API key
docker compose up --build
# API docs:      http://localhost:8000/docs
# Frontend:      http://localhost:3000
```

Or run the backend directly:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Project structure

```
veritas-rag/
├── backend/            FastAPI app (ingestion + RAG engine)
│   └── app/
│       ├── api/         routes: documents, query
│       ├── core/        chunking, parsing
│       ├── providers/   embeddings + chat (fake / OpenAI)
│       ├── vectorstore/ in-memory + pgvector backends
│       └── services/    ingestion, rag
├── frontend/           Next.js chat UI (coming in Phase 1)
├── evals/              golden set + evaluation runner
├── docker-compose.yml
└── ARCHITECTURE.md
```

---

## Author

**Nicolae Cristian Lapușneanu** — [LinkedIn](https://www.linkedin.com/in/nicolae-lapusneanu/) ·
[GitHub](https://github.com/LapusneanuNicolae) · niculapusneanu1@gmail.com
