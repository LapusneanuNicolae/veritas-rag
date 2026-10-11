# Evaluation

A non-deterministic system can't be improved by vibes. This folder holds the **golden set** and the
evaluation runner used to judge every change to chunking, retrieval or prompts.

## Metrics (RAGAS-style)
- **Faithfulness** — is the answer supported by the retrieved context? (anti-hallucination)
- **Answer relevancy** — does the answer actually address the question?
- **Context precision / recall** — did retrieval surface the right chunks?

## Files
- `golden_set.example.json` — sample question/answer pairs (copy to `golden_set.json` and extend)
- `run_evals.py` — runner (TODO Phase 3): executes the golden set, prints a scores table,
  and supports comparing two configs (e.g. with vs. without reranking).

## Usage (Phase 3)
```bash
python evals/run_evals.py --config default
```
