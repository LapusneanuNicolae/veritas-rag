"""Document ingestion endpoints: upload a file, track processing status.

Flow: upload -> parse -> chunk -> embed -> store in pgvector.
In Phase 1 this runs inline; it will move to async workers in a later phase.
"""
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.chunking import chunk_text
from app.core.parsing import extract_text

router = APIRouter()

# In-memory status store for the skeleton; replaced by a DB table in Phase 1.
_DOCUMENTS: dict[str, dict] = {}

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    doc_id = str(uuid.uuid4())
    raw = await file.read()

    # 1) parse -> text
    text = extract_text(raw, ext)
    # 2) chunk
    chunks = chunk_text(text)
    # 3) TODO(Phase 1): embed each chunk and persist to pgvector
    #    from app.core.embeddings import embed_texts
    #    from app.db.vectorstore import store_chunks
    #    store_chunks(doc_id, chunks, embed_texts([c.text for c in chunks]))

    _DOCUMENTS[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "status": "parsed",      # -> "embedded" once persistence lands
        "chunks": len(chunks),
    }
    return _DOCUMENTS[doc_id]


@router.get("/{doc_id}")
def get_document(doc_id: str) -> dict:
    doc = _DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("")
def list_documents() -> list[dict]:
    return list(_DOCUMENTS.values())
