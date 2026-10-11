"""Central configuration, loaded from environment variables.

Defaults are chosen so the app runs end-to-end with NO external services or API keys
(fake provider + in-memory vector store) — ideal for local dev, tests and CI.
Switch to real providers/pgvector via environment variables for production.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Veritas RAG"
    environment: str = "development"

    # --- Providers (model-agnostic) ---
    # llm_provider: "fake" | "openai"  (fake needs no API key)
    llm_provider: str = "fake"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_api_key: str = ""
    embedding_dim: int = 256  # used by the fake embedder / in-memory store

    # --- Vector backend: "memory" | "pgvector" ---
    vector_backend: str = "memory"
    database_url: str = "postgresql+psycopg://veritas:veritas@localhost:5432/veritas"
    redis_url: str = "redis://localhost:6379/0"

    # --- Retrieval tuning ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 20          # candidates retrieved (per source) before reranking
    top_n: int = 5           # chunks sent to the LLM after reranking

    # --- Retrieval strategy (Phase 2) ---
    use_hybrid: bool = True          # vector + keyword search fused with RRF
    use_rerank: bool = True          # rerank fused candidates down to top_n
    use_query_rewrite: bool = False  # rewrite the query before retrieval
    rerank_provider: str = "fake"    # "fake" | "cross-encoder"

    # --- Auth & production (Phase 5) ---
    jwt_secret: str = "dev-secret-change-me"
    jwt_expire_minutes: int = 1440
    require_auth: bool = False          # when False, requests use the "default" tenant
    rate_limit_per_minute: int = 120    # per client IP
    cache_enabled: bool = True
    cache_backend: str = "memory"       # "memory" | "redis"
    cache_ttl_seconds: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
