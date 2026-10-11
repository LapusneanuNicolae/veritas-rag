"""Answer cache. In-memory by default (TTL), optional Redis backend. Caching answers
avoids repeat LLM calls for identical questions — real cost/latency savings."""
from __future__ import annotations

import json
import time
from functools import lru_cache

from app.config import get_settings


class InMemoryCache:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        item = self._store.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.time():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str) -> None:
        self._store[key] = (time.time() + self.ttl, value)

    def clear(self) -> None:
        self._store.clear()


class RedisCache:
    def __init__(self, url: str, ttl: int):
        import redis  # lazy
        self._r = redis.Redis.from_url(url)
        self.ttl = ttl

    def get(self, key: str) -> str | None:
        val = self._r.get(key)
        return val.decode() if val else None

    def set(self, key: str, value: str) -> None:
        self._r.setex(key, self.ttl, value)

    def clear(self) -> None:
        self._r.flushdb()


@lru_cache
def get_cache():
    s = get_settings()
    if s.cache_backend == "redis":
        return RedisCache(s.redis_url, s.cache_ttl_seconds)
    return InMemoryCache(s.cache_ttl_seconds)


def make_key(tenant: str, question: str, top_n: int | None) -> str:
    return f"ans:{tenant}:{top_n}:{question.strip().lower()}"


def get_cached(tenant: str, question: str, top_n: int | None):
    if not get_settings().cache_enabled:
        return None
    raw = get_cache().get(make_key(tenant, question, top_n))
    return json.loads(raw) if raw else None


def set_cached(tenant: str, question: str, top_n: int | None, payload: dict) -> None:
    if not get_settings().cache_enabled:
        return
    get_cache().set(make_key(tenant, question, top_n), json.dumps(payload))
