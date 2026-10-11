"""Provider interfaces. Embeddings and chat sit behind these so the rest of the
app never imports a vendor SDK directly — switching provider is a config change."""
from __future__ import annotations

from typing import Iterator, Protocol


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class ChatProvider(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...

    def stream(self, system: str, user: str) -> Iterator[str]:
        ...
