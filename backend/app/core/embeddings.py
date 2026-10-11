"""Model-agnostic embeddings.

The provider is selected via config (openai / gemini / anthropic). Only the configured
provider's SDK is called at runtime, so switching is a config change, not a rewrite.
"""
from app.config import get_settings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return an embedding vector for each input text."""
    settings = get_settings()

    if settings.llm_provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.llm_api_key)
        resp = client.embeddings.create(model=settings.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    # TODO: gemini / anthropic providers (same interface)
    raise NotImplementedError(f"Embedding provider '{settings.llm_provider}' not implemented yet")
