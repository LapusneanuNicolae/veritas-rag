"""Lightweight per-query observability: what was retrieved, latency, and token cost.

In production this would feed LangSmith / Phoenix; here it's an in-process ring buffer
exposed via /api/traces so the behaviour (and cost) of every query is inspectable.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field

# approx USD per 1K tokens (input, output) — extend as needed
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.0025, 0.010),
    "fake": (0.0, 0.0),
}

_RECENT: "deque[dict]" = deque(maxlen=200)


def estimate_tokens(text: str) -> int:
    # rough heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def estimate_cost(model: str, prompt: str, completion: str) -> float:
    in_price, out_price = _PRICES.get(model, (0.0, 0.0))
    return round(
        estimate_tokens(prompt) / 1000 * in_price
        + estimate_tokens(completion) / 1000 * out_price,
        6,
    )


@dataclass
class QueryTrace:
    question: str
    rewritten_query: str
    provider: str
    model: str
    n_chunks: int
    documents: list[str]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    ts: float = field(default_factory=time.time)


def record(trace: QueryTrace) -> None:
    _RECENT.appendleft(asdict(trace))


def recent(limit: int = 50) -> list[dict]:
    return list(_RECENT)[:limit]


def stats() -> dict:
    traces = list(_RECENT)
    if not traces:
        return {"queries": 0, "total_cost_usd": 0.0, "avg_latency_ms": 0.0}
    return {
        "queries": len(traces),
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 6),
        "avg_latency_ms": round(sum(t["latency_ms"] for t in traces) / len(traces), 2),
    }
