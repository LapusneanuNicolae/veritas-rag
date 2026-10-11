"""Query transformation: rewrite a user question into a cleaner search query, and
decompose complex questions into sub-questions. Both improve retrieval recall."""
from __future__ import annotations

from app.providers.factory import get_chat_provider

_REWRITE_SYS = (
    "Rewrite the user's question into a concise, keyword-rich search query. "
    "Return only the rewritten query, nothing else."
)
_DECOMPOSE_SYS = (
    "Break the question into 2-4 simpler standalone sub-questions, one per line. "
    "If it is already simple, return it unchanged."
)


def rewrite_query(question: str) -> str:
    try:
        out = get_chat_provider().complete(_REWRITE_SYS, question).strip()
        return out or question
    except Exception:
        return question


def decompose_query(question: str) -> list[str]:
    try:
        out = get_chat_provider().complete(_DECOMPOSE_SYS, question)
        subs = [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()]
        return subs or [question]
    except Exception:
        return [question]
