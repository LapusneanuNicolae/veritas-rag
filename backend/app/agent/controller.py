"""Lightweight agentic RAG controller.

Instead of a single retrieve→answer pass, the agent:
  1. decomposes a complex question into sub-questions,
  2. retrieves for each and merges unique chunks,
  3. generates an answer,
  4. self-checks faithfulness and self-corrects (broadens retrieval) if it's low.

Implemented as an explicit loop (no heavy framework) so it stays transparent and
testable; the `steps` it returns make the agent's reasoning inspectable in the UI.
"""
from __future__ import annotations

from app.config import get_settings
from app.core.query_transform import decompose_query
from app.eval.metrics import faithfulness
from app.providers.factory import get_chat_provider
from app.services import rag
from app.types import RetrievedChunk

FAITHFULNESS_THRESHOLD = 0.5


def _merge(chunk_lists: list[list[RetrievedChunk]], top_n: int) -> list[RetrievedChunk]:
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for lst in chunk_lists:
        for c in lst:
            key = c.chunk_id or f"{c.document}|{c.location}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
    return merged[:top_n]


def agentic_answer(question: str, tenant_id: str = "default") -> dict:
    s = get_settings()
    steps: list[dict] = []

    sub_questions = decompose_query(question)
    steps.append({"step": "decompose", "detail": f"{len(sub_questions)} sub-question(s)"})

    chunk_lists = [rag.retrieve(sub, tenant_id=tenant_id) for sub in sub_questions]
    merged = _merge(chunk_lists, s.top_n)
    steps.append({"step": "retrieve",
                  "detail": f"{len(merged)} unique chunks from {len(sub_questions)} query(ies)"})

    if not merged:
        steps.append({"step": "answer", "detail": "no context available"})
        return {"answer": "I don't have indexed context yet. Upload documents first.",
                "citations": [], "sub_questions": sub_questions, "steps": steps,
                "faithfulness": 0.0, "self_corrected": False}

    chat = get_chat_provider()
    prompt = rag.build_prompt(question, merged)
    text = chat.complete(rag.SYSTEM_PROMPT, prompt)
    score = faithfulness(text, [c.text for c in merged])
    steps.append({"step": "self_check", "detail": f"faithfulness={round(score, 3)}"})

    self_corrected = False
    if score < FAITHFULNESS_THRESHOLD:
        broadened = rag.retrieve(question, top_n=s.top_n * 2, tenant_id=tenant_id)
        if broadened:
            merged = broadened
            prompt = rag.build_prompt(question, merged)
            text = chat.complete(rag.SYSTEM_PROMPT, prompt)
            score = faithfulness(text, [c.text for c in merged])
            self_corrected = True
            steps.append({"step": "self_correct",
                          "detail": f"broadened retrieval, faithfulness={round(score, 3)}"})

    return {"answer": text, "citations": [c.as_citation() for c in merged],
            "sub_questions": sub_questions, "steps": steps,
            "faithfulness": round(score, 3), "self_corrected": self_corrected}
