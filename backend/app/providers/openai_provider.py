"""Real OpenAI-backed providers (lazy SDK import so the app runs without openai
installed when using the fake provider)."""
from __future__ import annotations

from typing import Iterator

from app.config import get_settings


class OpenAIEmbeddingProvider:
    def __init__(self):
        s = get_settings()
        self.model = s.embedding_model
        self.dim = 1536  # text-embedding-3-small
        self._api_key = s.llm_api_key

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client().embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class OpenAIChatProvider:
    def __init__(self):
        s = get_settings()
        self.model = s.llm_model
        self._api_key = s.llm_api_key

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key)

    def complete(self, system: str, user: str) -> str:
        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        stream = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
