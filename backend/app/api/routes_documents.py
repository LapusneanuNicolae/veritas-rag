"""Document ingestion endpoints: upload a file, list/track documents."""
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingestion import ingest_document, SUPPORTED

router = APIRouter()

# Lightweight registry of ingested documents (metadata only).
_DOCUMENTS: dict[str, dict] = {}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    raw = await file.read()
    try:
        result = ingest_document(file.filename or "upload", raw, ext)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _DOCUMENTS[result["id"]] = result
    return result


@router.get("")
def list_documents() -> list[dict]:
    return list(_DOCUMENTS.values())


@router.get("/{doc_id}")
def get_document(doc_id: str) -> dict:
    doc = _DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
