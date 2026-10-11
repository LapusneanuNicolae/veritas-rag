"""Query endpoint: ask a question, get a grounded answer with citations.

Pipeline (see app/core/retrieval.py):
  rewrite -> hybrid retrieve -> rerank -> build context -> LLM -> cite
"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.retrieval import retrieve
from app.core.llm import generate_answer

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
    # 1) retrieve the most relevant chunks (hybrid + rerank — see retrieval.py)
    chunks = retrieve(req.question, top_n=req.top_n)

    # 2) generate a grounded answer; the LLM is told to answer only from context
    #    or to say it doesn't know.
    answer, citations = generate_answer(req.question, chunks)

    return AskResponse(
        answer=answer,
        citations=[Citation(**c) for c in citations],
    )
