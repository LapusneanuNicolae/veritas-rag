"""Model-agnostic answer generation.

Builds a grounded prompt from retrieved chunks and asks the LLM to answer ONLY from
that context (or say it doesn't know), then returns the answer plus citations.
"""
from app.config import get_settings

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't know. "
    "Cite the sources you used by their [n] markers."
)


def _build_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c.get('document')} @ {c.get('location')})\n{c.get('text')}")
    return "\n\n".join(lines)


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Return (answer, citations). Falls back to 'I don't know' with no context."""
    if not chunks:
        return (
            "I don't have any indexed context yet, so I can't answer this reliably. "
            "Upload documents first.",
            [],
        )

    settings = get_settings()
    context = _build_context(chunks)

    if settings.llm_provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.llm_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
        )
        answer = resp.choices[0].message.content or ""
    else:
        # TODO: gemini / anthropic providers (same interface)
        raise NotImplementedError(f"LLM provider '{settings.llm_provider}' not implemented yet")

    citations = [
        {"document": c.get("document", ""), "location": c.get("location", ""),
         "snippet": (c.get("text", "")[:160] + "…")}
        for c in chunks
    ]
    return answer, citations
