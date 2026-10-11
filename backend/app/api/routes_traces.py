"""Observability endpoints: recent query traces and aggregate stats."""
from fastapi import APIRouter

from app.observability import trace as obs

router = APIRouter()


@router.get("/traces")
def list_traces(limit: int = 50) -> list[dict]:
    return obs.recent(limit=limit)


@router.get("/stats")
def get_stats() -> dict:
    return obs.stats()
