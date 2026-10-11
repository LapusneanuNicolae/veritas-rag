"""Veritas RAG — FastAPI entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings
from app.api import routes_documents, routes_query, routes_traces

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Production-grade, model-agnostic RAG with verifiable citations.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(routes_query.router, prefix="/api", tags=["query"])
app.include_router(routes_traces.router, prefix="/api", tags=["observability"])


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": __version__}
