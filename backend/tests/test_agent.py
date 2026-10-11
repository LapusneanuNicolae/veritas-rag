"""Phase 4 tests: agentic controller."""
from fastapi.testclient import TestClient

from app.agent.controller import agentic_answer
from app.main import app
from app.services.ingestion import ingest_document
from app.vectorstore.factory import get_vectorstore

client = TestClient(app)


def setup_function():
    get_vectorstore().clear()


def test_agentic_answer_structure():
    ingest_document("policy.txt",
                    b"Refunds are available within 30 days of purchase.", ".txt")
    result = agentic_answer("What is the refund window?")
    assert result["answer"]
    assert result["citations"]
    assert result["sub_questions"]
    # the agent should log its reasoning steps
    kinds = [s["step"] for s in result["steps"]]
    assert "decompose" in kinds and "retrieve" in kinds and "self_check" in kinds


def test_agentic_endpoint():
    ingest_document("doc.txt", b"Veritas RAG is a RAG platform with citations.", ".txt")
    resp = client.post("/api/ask/agentic", json={"question": "What is Veritas RAG?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and "steps" in body and "faithfulness" in body


def test_agentic_without_documents():
    result = agentic_answer("anything?")
    assert result["citations"] == []
    assert result["faithfulness"] == 0.0
