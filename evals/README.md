# Evaluation

A non-deterministic system can't be improved by vibes. This folder holds the **golden set** and the
evaluation runner used to judge every change to chunking, retrieval or prompts.

## Metrics (RAGAS-style)
- **Faithfulness** — is the answer supported by the retrieved context? (anti-hallucination)
- **Answer relevancy** — does the answer actually address the question?
- **Context precision / recall** — did retrieval surface the right chunks?

## Files
- `golden_set.example.json` — sample question/answer pairs (copy to `golden_set.json` and extend)
- `run_evals.py` — runner (implemented): seeds a demo corpus, executes the golden set and prints a
  scores table. Metric logic lives in `backend/app/eval/`.

## Usage
```bash
python evals/run_evals.py                 # uses golden_set.example.json
python evals/run_evals.py --golden my.json --label with-rerank
```

Example output:
```
=== Eval report: default  (3 cases) ===
  faithfulness         1.0
  answer_relevancy     0.456
  context_precision    0.333
  context_recall       0.667
  citation_accuracy    1.0
```

> The metrics use embedding- and lexical-based approximations so they run deterministically in CI.
> The scoring backend is swappable for an LLM judge (RAGAS) without changing callers.
