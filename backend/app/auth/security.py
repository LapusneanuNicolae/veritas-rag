"""Minimal, dependency-free auth: HS256 JWTs and PBKDF2 password hashing.

Implemented with the standard library (hmac/hashlib) to avoid extra dependencies while
staying standards-compatible. Swap for PyJWT/passlib in a larger deployment if preferred.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from app.config import get_settings


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


# --- passwords ---

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 100_000)
    return hmac.compare_digest(digest.hex(), digest_hex)


# --- JWT (HS256) ---

def create_access_token(sub: str, tenant_id: str) -> str:
    s = get_settings()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": sub,
        "tenant_id": tenant_id,
        "exp": int(time.time()) + s.jwt_expire_minutes * 60,
    }
    signing_input = f"{_b64e(json.dumps(header).encode())}.{_b64e(json.dumps(payload).encode())}"
    sig = hmac.new(s.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64e(sig)}"


def decode_token(token: str) -> dict:
    s = get_settings()
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise ValueError("Malformed token") from exc
    signing_input = f"{header_b64}.{payload_b64}"
    expected = hmac.new(s.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64d(sig_b64)):
        raise ValueError("Invalid signature")
    payload = json.loads(_b64d(payload_b64))
    if payload.get("exp", 0) < int(time.time()):
        raise ValueError("Token expired")
    return payload
