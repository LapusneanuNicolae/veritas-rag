"""Central configuration, loaded from environment variables.

Keeping all settings in one typed object makes the app easy to configure and test,
and keeps secrets out of the code.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Veritas RAG"
    environment: str = "development"

    # --- LLM provider (model-agnostic) ---
    # one of: "openai" | "gemini" | "anthropic"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_api_key: str = ""

    # --- Infrastructure ---
    database_url: str = "postgresql+psycopg://veritas:veritas@localhost:5432/veritas"
    redis_url: str = "redis://localhost:6379/0"

    # --- Retrieval tuning ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 20          # candidates retrieved before reranking
    top_n: int = 5           # chunks sent to the LLM after reranking


@lru_cache
def get_settings() -> Settings:
    return Settings()
