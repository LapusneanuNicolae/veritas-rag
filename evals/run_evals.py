#!/usr/bin/env python
"""CLI: run the golden-set evaluation and print a scores table.

    python evals/run_evals.py                 # uses evals/golden_set.example.json
    python evals/run_evals.py --golden my.json --docs ./corpus

Seeds the (in-memory) store with the demo corpus first so the pipeline has something
to retrieve, then scores faithfulness / answer relevancy / context precision & recall.
"""
import argparse
import os
import sys

# allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.eval.runner import load_golden_set, run_evals  # noqa: E402
from app.services.ingestion import ingest_document  # noqa: E402

DEMO_DOCS = {
    "refund-policy.pdf": b"Refunds are available within 30 days of purchase. "
                         b"After 30 days, purchases are final.",
    "handbook.docx": b"The document owner is the Finance department. "
                     b"Questions should be directed to finance@example.com.",
}


def seed_demo():
    for name, content in DEMO_DOCS.items():
        ext = "." + name.rsplit(".", 1)[-1]
        # store as text regardless of extension for the demo seeder
        ingest_document(name, content, ".txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=os.path.join(
        os.path.dirname(__file__), "golden_set.example.json"))
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    seed_demo()
    report = run_evals(load_golden_set(args.golden), label=args.label)

    print(f"\n=== Eval report: {report['label']}  ({report['n_cases']} cases) ===")
    agg = report["aggregate"]
    for k, v in agg.items():
        print(f"  {k:<20} {v}")
    print("\nPer-case:")
    for r in report["cases"]:
        print(f"  - faith={r['faithfulness']}  rel={r['answer_relevancy']}  "
              f"ctx_p={r['context_precision']}  ctx_r={r['context_recall']}  "
              f"cite={'OK' if r['citation_ok'] else 'MISS'}  | {r['question'][:50]}")
    print()


if __name__ == "__main__":
    main()
