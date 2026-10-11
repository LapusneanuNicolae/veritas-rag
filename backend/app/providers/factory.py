"""Select providers based on configuration."""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.providers.base import ChatProvider, EmbeddingProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    s = get_settings()
    if s.llm_provider == "fake":
        from app.providers.fake import FakeEmbeddingProvider
        return FakeEmbeddingProvider(dim=s.embedding_dim)
    if s.llm_provider == "openai":
        from app.providers.openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider()
    raise ValueError(f"Unknown llm_provider: {s.llm_provider}")


@lru_cache
def get_chat_provider() -> ChatProvider:
    s = get_settings()
    if s.llm_provider == "fake":
        from app.providers.fake import FakeChatProvider
        return FakeChatProvider()
    if s.llm_provider == "openai":
        from app.providers.openai_provider import OpenAIChatProvider
        return OpenAIChatProvider()
    raise ValueError(f"Unknown llm_provider: {s.llm_provider}")
