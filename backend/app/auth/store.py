"""In-memory user store (demo). Each user belongs to a tenant; swap for a DB table
in production — the interface stays the same."""
from __future__ import annotations

import uuid

from app.auth.security import hash_password, verify_password

_USERS: dict[str, dict] = {}


def create_user(email: str, password: str) -> dict:
    if email in _USERS:
        raise ValueError("User already exists")
    user = {"email": email, "password": hash_password(password),
            "tenant_id": str(uuid.uuid4())}
    _USERS[email] = user
    return user


def authenticate(email: str, password: str) -> dict | None:
    user = _USERS.get(email)
    if user and verify_password(password, user["password"]):
        return user
    return None
