"""pgvector-backed store (production). Dense search via the `<=>` cosine operator,
keyword search via PostgreSQL full-text search. SDK imported lazily so the fake /
in-memory path has no DB dependency."""
from __future__ import annotations

from app.config import get_settings
from app.types import RetrievedChunk, StoredChunk


class PgVectorStore:
    def __init__(self):
        from sqlalchemy import create_engine, text

        s = get_settings()
        self._text = text
        self._engine = create_engine(s.database_url, future=True)
        self._dim = get_settings().embedding_dim if s.llm_provider == "fake" else 1536
        self._init_schema()

    def _init_schema(self) -> None:
        text = self._text
        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id   TEXT PRIMARY KEY,
                    tenant_id  TEXT NOT NULL DEFAULT 'default',
                    document   TEXT NOT NULL,
                    location   TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    embedding  vector({self._dim}),
                    content_tsv tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (content_tsv)"))

    def add(self, chunks: list[StoredChunk]) -> None:
        text = self._text
        with self._engine.begin() as conn:
            for c in chunks:
                conn.execute(text("""
                    INSERT INTO chunks (chunk_id, tenant_id, document, location, content, embedding)
                    VALUES (:id, :tenant, :doc, :loc, :content, :emb)
                    ON CONFLICT (chunk_id) DO NOTHING
                """), {"id": c.chunk_id, "tenant": c.tenant_id, "doc": c.document,
                       "loc": c.location, "content": c.text, "emb": str(c.embedding)})

    def vector_search(self, embedding: list[float], top_k: int,
                      tenant_id: str = "default") -> list[RetrievedChunk]:
        text = self._text
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT document, location, content, chunk_id,
                       1 - (embedding <=> :emb) AS score
                FROM chunks
                WHERE tenant_id = :tenant
                ORDER BY embedding <=> :emb
                LIMIT :k
            """), {"emb": str(embedding), "tenant": tenant_id, "k": top_k}).fetchall()
        return [RetrievedChunk(text=r.content, document=r.document, location=r.location,
                               score=float(r.score), chunk_id=r.chunk_id) for r in rows]

    def keyword_search(self, query: str, top_k: int,
                       tenant_id: str = "default") -> list[RetrievedChunk]:
        text = self._text
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT document, location, content, chunk_id,
                       ts_rank(content_tsv, plainto_tsquery('english', :q)) AS score
                FROM chunks
                WHERE tenant_id = :tenant
                  AND content_tsv @@ plainto_tsquery('english', :q)
                ORDER BY score DESC
                LIMIT :k
            """), {"q": query, "tenant": tenant_id, "k": top_k}).fetchall()
        return [RetrievedChunk(text=r.content, document=r.document, location=r.location,
                               score=float(r.score), chunk_id=r.chunk_id) for r in rows]

    def count(self, tenant_id: str | None = None) -> int:
        text = self._text
        with self._engine.connect() as conn:
            if tenant_id is None:
                return conn.execute(text("SELECT count(*) FROM chunks")).scalar() or 0
            return conn.execute(text("SELECT count(*) FROM chunks WHERE tenant_id = :t"),
                                {"t": tenant_id}).scalar() or 0

    def clear(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._text("TRUNCATE chunks"))
