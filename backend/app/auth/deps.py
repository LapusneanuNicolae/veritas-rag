"""FastAPI dependency that resolves the caller's tenant from a bearer token.

If auth isn't required (default dev mode) and no valid token is present, requests fall
back to the shared "default" tenant so the app is usable without logging in."""
from __future__ import annotations

from fastapi import Header, HTTPException

from app.auth.security import decode_token
from app.config import get_settings


def current_tenant(authorization: str | None = Header(default=None)) -> str:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]

    if token:
        try:
            return decode_token(token)["tenant_id"]
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc))

    if get_settings().require_auth:
        raise HTTPException(status_code=401, detail="Authentication required")
    return "default"
