"""Query endpoints: ask (grounded answer with citations) and ask/stream (SSE)."""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.controller import agentic_answer
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


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    text, citations = rag.answer(req.question, top_n=req.top_n)
    return AskResponse(answer=text, citations=[Citation(**c) for c in citations])


@router.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    def event_stream():
        for token in rag.answer_stream(req.question, top_n=req.top_n):
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
def ask_agentic(req: AskRequest) -> AgenticResponse:
    """Multi-step agentic answer: decompose → retrieve → answer → self-correct."""
    result = agentic_answer(req.question)
    return AgenticResponse(**result)
