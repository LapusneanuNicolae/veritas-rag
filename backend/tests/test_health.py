from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_without_documents_returns_graceful_answer():
    resp = client.post("/api/ask", json={"question": "What is this about?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["citations"] == []
