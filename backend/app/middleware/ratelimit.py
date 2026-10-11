"""Simple in-memory sliding-window rate limiter (per client IP).

Good enough for a single instance; back it with Redis for multi-instance deployments."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import get_settings

_HITS: dict[str, deque] = defaultdict(deque)
_WINDOW = 60.0


async def rate_limit_middleware(request: Request, call_next):
    limit = get_settings().rate_limit_per_minute
    client = request.client.host if request.client else "unknown"
    now = time.time()
    hits = _HITS[client]
    while hits and hits[0] < now - _WINDOW:
        hits.popleft()
    if len(hits) >= limit:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    hits.append(now)
    return await call_next(request)
