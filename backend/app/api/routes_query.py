"""Query endpoints: ask (cached, tenant-scoped), streaming, and agentic."""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import cache
from app.agent.controller import agentic_answer
from app.auth.deps import current_tenant
from app.services import rag

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    top_n: int | None = None


class Citation(BaseModel):
    document: str
    location: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cached: bool = False


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, tenant: str = Depends(current_tenant)) -> AskResponse:
    hit = cache.get_cached(tenant, req.question, req.top_n)
    if hit:
        return AskResponse(answer=hit["answer"],
                           citations=[Citation(**c) for c in hit["citations"]], cached=True)
    text, citations = rag.answer(req.question, top_n=req.top_n, tenant_id=tenant)
    cache.set_cached(tenant, req.question, req.top_n,
                     {"answer": text, "citations": citations})
    return AskResponse(answer=text, citations=[Citation(**c) for c in citations])


@router.post("/ask/stream")
def ask_stream(req: AskRequest, tenant: str = Depends(current_tenant)) -> StreamingResponse:
    def event_stream():
        for token in rag.answer_stream(req.question, top_n=req.top_n, tenant_id=tenant):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class AgenticResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sub_questions: list[str]
    steps: list[dict]
    faithfulness: float
    self_corrected: bool


@router.post("/ask/agentic", response_model=AgenticResponse)
def ask_agentic(req: AskRequest, tenant: str = Depends(current_tenant)) -> AgenticResponse:
    """Multi-step agentic answer: decompose → retrieve → answer → self-correct."""
    result = agentic_answer(req.question, tenant_id=tenant)
    return AgenticResponse(**result)
