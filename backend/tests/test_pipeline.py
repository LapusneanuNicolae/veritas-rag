"""End-to-end Phase 1 tests using the fake provider + in-memory store (no external deps)."""
from fastapi.testclient import TestClient

from app.main import app
from app.vectorstore.factory import get_vectorstore

client = TestClient(app)


def setup_function():
    get_vectorstore().clear()


def _upload(text: str, name: str = "doc.txt"):
    return client.post("/api/documents/upload",
                       files={"file": (name, text.encode(), "text/plain")})


def test_upload_embeds_chunks():
    resp = _upload("The refund window is 30 days from purchase.")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "embedded"
    assert body["chunks"] >= 1


def test_ask_returns_grounded_answer_with_citations():
    _upload("The capital of France is Paris. Paris is known for the Eiffel Tower.")
    _upload("Refunds are available within 30 days of purchase.", name="policy.txt")

    resp = client.post("/api/ask", json={"question": "What is the refund window?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert len(body["citations"]) >= 1
    # the policy document should be retrieved for a refund question
    docs = [c["document"] for c in body["citations"]]
    assert "policy.txt" in docs


def test_ask_without_documents_declines():
    resp = client.post("/api/ask", json={"question": "anything?"})
    assert resp.status_code == 200
    assert resp.json()["citations"] == []


def test_stream_endpoint():
    _upload("Veritas RAG is a retrieval-augmented generation platform.")
    with client.stream("POST", "/api/ask/stream",
                       json={"question": "What is Veritas RAG?"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "[DONE]" in body
