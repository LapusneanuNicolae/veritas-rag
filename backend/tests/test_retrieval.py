"""Phase 2 tests: hybrid search, RRF fusion, reranking."""
from app.core.fusion import reciprocal_rank_fusion
from app.core.rerank import FakeReranker
from app.services import rag
from app.types import RetrievedChunk
from app.vectorstore.factory import get_vectorstore


def _rc(doc, score, text="", cid=""):
    return RetrievedChunk(text=text or doc, document=doc, location="chunk 1",
                          score=score, chunk_id=cid or doc)


def test_rrf_merges_and_ranks():
    list_a = [_rc("A", 0.9), _rc("B", 0.8)]
    list_b = [_rc("B", 0.5), _rc("C", 0.4)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    docs = [c.document for c in fused]
    # B appears in both lists -> should rank first after fusion
    assert docs[0] == "B"
    assert set(docs) == {"A", "B", "C"}


def test_fake_reranker_prefers_lexical_overlap():
    chunks = [
        _rc("doc1", 0.5, text="the mitochondria is the powerhouse of the cell"),
        _rc("doc2", 0.9, text="paris is the capital of france"),
    ]
    out = FakeReranker().rerank("what is the capital of france", chunks, top_n=2)
    assert out[0].document == "doc2"  # higher lexical overlap wins despite lower prior score


def test_hybrid_retrieval_end_to_end():
    store = get_vectorstore()
    store.clear()
    rag.retrieve  # ensure import
    from app.services.ingestion import ingest_document
    ingest_document("space.txt", b"The sun is a star at the center of the solar system.", ".txt")
    ingest_document("refund.txt", b"Customers can request a refund within 30 days.", ".txt")

    results = rag.retrieve("refund window", top_n=3)
    assert results
    assert results[0].document == "refund.txt"
