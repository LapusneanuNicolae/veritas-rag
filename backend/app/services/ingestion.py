"""Ingestion service: parse → chunk → embed → store."""
from __future__ import annotations

import uuid

from app.core.chunking import chunk_text
from app.core.parsing import extract_text
from app.providers.factory import get_embedding_provider
from app.types import StoredChunk
from app.vectorstore.factory import get_vectorstore

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


def ingest_document(filename: str, raw: bytes, ext: str,
                    tenant_id: str = "default") -> dict:
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")

    doc_id = str(uuid.uuid4())
    text = extract_text(raw, ext)
    chunks = chunk_text(text, document=filename)
    if not chunks:
        return {"id": doc_id, "filename": filename, "status": "empty", "chunks": 0}

    embedder = get_embedding_provider()
    vectors = embedder.embed([c.text for c in chunks])

    stored = [
        StoredChunk(
            chunk_id=f"{doc_id}:{c.index}",
            document=filename,
            location=c.location,
            text=c.text,
            embedding=vec,
            tenant_id=tenant_id,
        )
        for c, vec in zip(chunks, vectors)
    ]
    get_vectorstore().add(stored)

    return {"id": doc_id, "filename": filename, "status": "embedded",
            "chunks": len(stored)}
