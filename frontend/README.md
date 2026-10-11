# Frontend (Next.js)

Chat UI for Veritas RAG: upload documents, ask questions, see answers with inline
source citations, and toggle **agentic mode** (decomposition + self-correction, with
the agent's reasoning steps shown).

## Run

```bash
cp .env.local.example .env.local      # point NEXT_PUBLIC_API_URL at the backend
npm install
npm run dev                           # http://localhost:3000
```

The backend must be running (see repo root README). With the default settings the API
is at http://localhost:8000.
