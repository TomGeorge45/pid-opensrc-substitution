# All VLM Stages — Benchmark Harness

One parametrized notebook (`AllStages_Benchmark.ipynb`) benchmarks every VLM stage except
Stage 4 (which has its own dedicated harness in `notebooks/stage4/`). Set `MODEL` in cell 1,
Run All. Each run appends to a shared results CSV on the private HF dataset repo
(`benchmarks/all_vlm_stages_results.csv`), so cross-model comparison accumulates across runs.

## Why one notebook instead of one per model

Per-model copies drift: a fix applied to one silently doesn't reach the others, and then
cross-model numbers aren't comparable. The model adapter layer (cell 4) reduces every model
to `generate(image, prompt) -> text`; the stage sections never know which model is running.
To add a model, add one loader to `LOADERS` — nothing else changes.

## Supported models

`gpt-5.5` (OpenAI API, low reasoning — incumbent reference) · `qwen-base` · `qwen-domain-v1`
· `qwen-domain-v2` (our HF adapters) · `molmo2`

## Custom / single-stage tests

Set `MODEL`, run cells 1–5 (setup + model + results plumbing), then run only the stage
section you want. Results still record to the same CSV.

## Honesty table — what each score means

| Stage | GT | Caveat |
|---|---|---|
| 1 Sheet classification | none | Recall-only on known-P&IDs; yes-to-everything scores 100% |
| 2 Title block | none | Raw extractions stored; only cross-model agreement is computable |
| 5 OCR tiebreaker | synthesized | Real accuracy, but pseudo-GT from Tesseract-confident words |
| 10.5 Skid grouping | sparse | Not benchmarked yet; stage 12 is the nearest proxy |
| 12 Relation validation | **real** (PID2Graph) | Most trustworthy number in the harness |
| 13 Entity validation | partial (Gupta) | Existence-only; type-correctness untestable without typed real GT |

Never average across stages. Quote each with its caveat attached.
