"""Evaluation runner: execute a golden set through the RAG pipeline and score it.

Usage (programmatic):
    from app.eval.runner import run_evals, load_golden_set
    report = run_evals(load_golden_set("evals/golden_set.json"))

A non-deterministic system can't be improved by vibes — this gives a repeatable score
so any change to chunking / retrieval / prompts can be compared on the same set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from app.eval import metrics
from app.services import rag


@dataclass
class EvalCase:
    question: str
    reference_answer: str
    must_cite: list[str]


def load_golden_set(path: str) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [EvalCase(d["question"], d.get("reference_answer", ""),
                     d.get("must_cite", [])) for d in data]


def evaluate_case(case: EvalCase) -> dict:
    answer, chunks = rag.answer_with_context(case.question)
    contexts = [c.text for c in chunks]
    cited_docs = {c.document for c in chunks}
    citation_ok = all(doc in cited_docs for doc in case.must_cite) if case.must_cite else True
    return {
        "question": case.question,
        "faithfulness": round(metrics.faithfulness(answer, contexts), 3),
        "answer_relevancy": round(metrics.answer_relevancy(case.question, answer), 3),
        "context_precision": round(metrics.context_precision(contexts, case.reference_answer), 3),
        "context_recall": round(metrics.context_recall(contexts, case.reference_answer), 3),
        "citation_ok": citation_ok,
    }


def run_evals(cases: list[EvalCase], label: str = "default") -> dict:
    rows = [evaluate_case(c) for c in cases]
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    aggregate = {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys} if rows else {}
    aggregate["citation_accuracy"] = (
        round(sum(1 for r in rows if r["citation_ok"]) / len(rows), 3) if rows else 0.0)
    return {"label": label, "n_cases": len(rows), "aggregate": aggregate, "cases": rows}
