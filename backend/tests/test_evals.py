"""Phase 3 tests: evaluation metrics + runner + observability."""
from app.eval import metrics
from app.eval.runner import EvalCase, run_evals
from app.observability import trace as obs
from app.services.ingestion import ingest_document
from app.vectorstore.factory import get_vectorstore


def setup_function():
    get_vectorstore().clear()


def test_faithfulness_rewards_grounded_answers():
    ctx = ["Paris is the capital of France."]
    grounded = metrics.faithfulness("The capital of France is Paris.", ctx)
    hallucinated = metrics.faithfulness("The capital of France is Berlin and Tokyo.", ctx)
    assert grounded > hallucinated


def test_context_recall_and_precision():
    contexts = ["Refunds are available within 30 days of purchase."]
    assert metrics.context_recall(contexts, "refund within 30 days") > 0.5
    assert metrics.context_precision(contexts, "refund within 30 days") == 1.0


def test_run_evals_produces_aggregate():
    ingest_document("refund.txt", b"Refunds are available within 30 days of purchase.", ".txt")
    cases = [EvalCase("What is the refund window?",
                      "Refunds are available within 30 days.", ["refund.txt"])]
    report = run_evals(cases)
    assert report["n_cases"] == 1
    assert "faithfulness" in report["aggregate"]
    assert report["aggregate"]["citation_accuracy"] == 1.0


def test_tracing_records_query_cost():
    ingest_document("doc.txt", b"Veritas RAG is a retrieval augmented generation platform.", ".txt")
    from app.services import rag
    rag.answer("What is Veritas RAG?")
    traces = obs.recent()
    assert traces
    assert traces[0]["n_chunks"] >= 1
    assert "cost_usd" in traces[0]
