"""Document ingestion endpoints: upload a file, list/track documents (tenant-scoped)."""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from app.auth.deps import current_tenant
from app.services.ingestion import ingest_document, SUPPORTED

router = APIRouter()

# Lightweight registry of ingested documents (metadata only), per tenant.
_DOCUMENTS: dict[str, dict] = {}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...),
                         tenant: str = Depends(current_tenant)) -> dict:
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    raw = await file.read()
    try:
        result = ingest_document(file.filename or "upload", raw, ext, tenant_id=tenant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result["tenant_id"] = tenant
    _DOCUMENTS[result["id"]] = result
    return result


@router.get("")
def list_documents(tenant: str = Depends(current_tenant)) -> list[dict]:
    return [d for d in _DOCUMENTS.values() if d.get("tenant_id") == tenant]


@router.get("/{doc_id}")
def get_document(doc_id: str, tenant: str = Depends(current_tenant)) -> dict:
    doc = _DOCUMENTS.get(doc_id)
    if not doc or doc.get("tenant_id") != tenant:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
