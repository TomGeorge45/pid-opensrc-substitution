# Objective, constraints, and how the work was done

## 1. The objective

Replace every cloud model call in `pnid-intelligence-agent` — a working 15-stage P&ID
knowledge-graph agent — with a local counterpart, fine-tuned per stage, and prove equivalence.
The cloud models being replaced: `claude-sonnet-4-6`, `claude-opus-4-8`, Google Cloud Vision.

Of 16 stages, **9 were in scope** (1, 1.5, 2, 4, 5, 10.5, dynamic cropping, 12, 13) and 7 out of
scope (anything already pure Python / OpenCV / NetworkX / regex — those were never the problem).

The strategy was sequenced deliberately: **Stage 4 (symbol detection) goes first**, because it is
the only stage with full real ground truth, and its winner becomes the **shared base VLM** every
other stage reuses through prompting alone. One base, one task-neutral domain fine-tune, then a
small number of per-stage adapters only where measurement showed a need.

Three candidates: **Qwen3-VL 7B/8B**, **InternVL3-8B** (cleanest licence, MIT), **Molmo-7B**
(Apache-2.0, native pixel-pointing). Selection was to happen once, at Stage 4, on Gupta boxes.

## 2. The hard rules the project held itself to

These were written into `CLAUDE.md` at the outset and enforced throughout. They are worth recording
because several of them are the reason the project's negative findings are trustworthy.

1. **No trained-from-scratch detector architectures.** No YOLO, RT-DETR, U-Net, Siamese. Local VLMs,
   LLMs and OCR only. One documented exception: if the fine-tuned Stage-4 VLM couldn't reach a set
   fraction of the incumbent's recall, a dedicated detector could re-enter scope for Stage 4 alone.
2. **Real data judges.** Gupta and PID2Graph are real; Kaggle is synthetic. Synthetic was allowed as
   fine-tuning volume, never as a pass/fail bar.
3. **The two-part metric, never averaged.** Detection scored class-agnostically on Gupta (real);
   typing scored on Kaggle's 32 classes (synthetic). Reported side by side, always, with a coverage
   caveat attached to the typing half.
4. **Never recall alone as a pass bar.** mAP@0.5 or F1. A model emitting thousands of boxes scores
   near-perfect recall with garbage precision.
5. **Frozen test split**, leak-asserted before every training run.
6. **No MLflow** — `results.csv` plus per-experiment markdown.

**A known permanent limitation was logged at the start and never resolved:** find-and-type-correctly
*jointly* on real drawings is untestable, because no real typed ground truth exists anywhere. So
"detection 0.90 / typing 0.85" must never be read as "typed detection ≈ 0.85 on real sheets." It
isn't tested. For anyone. Including the incumbent.

## 3. Deliberate deviations from the spec

Three, all recorded rather than silent:

**Two bases, not one.** The spec commits to a single shared base chosen at Stage 4. In practice
**Molmo2 took Stage 4 and Qwen3-VL-8B became the shared base for every other VLM stage.** Grounds:
Qwen won the reasoning probe, and the spec itself anticipated reconsidering the base if the Stage-4
winner underperformed on reasoning. Cost: two model lifecycles instead of one.

**Scope grew past "Stage 4 only."** `CLAUDE.md` scopes the repo to Stage 4. The entity-extraction
multi-arm architecture and the entire relationship-stage effort go well beyond that. This was
directed, not accidental, but `CLAUDE.md` was never updated to reflect it — a documentation
inconsistency that persisted to the end.

**Training sequence resequenced.** The combined detection fine-tune was moved ahead of the
task-neutral domain pass, on an estimate (not a measurement) that layered training would cost
~17–25 h per model versus ~8.5–12 h. Reconciled with the one-base rule by branching the deferred
domain pass from untouched original weights — two checkpoints from one base rather than one polluted
checkpoint.

## 4. How the work was actually conducted

This matters for reading the record, because the project ran as **three sequential Claude Code
sessions**, not one continuous effort and not in parallel:

| Session | Dates | Commits | Focus |
|---|---|---|---|
| 1 | 2026-07-09 → ~07-13 | ~112 | Dataset integrity, agent-matched tiling, Stage-4 candidate bake-off, first domain-adaptation training runs, the dataset world-scan |
| 2 | ~07-14 → 07-25 | ~20 | Per-stage v3 adapters, all-stages benchmark harness, e2e cascade through the real agent, multi-arm entity extraction, relationship-stage Parts B–D |
| 3 | 07-27 → 07-28 | 4 | Pipeline 3 v2 — symbol-first relationship extraction, precision adjudication, close-out |

Continuity was maintained by **handover documents**: each session ended by writing a structured
write-up, and the next began by reading it. That produced the four dated session write-ups plus
`AI_Continuation_Document-27Jul2026-1500.md` and, eventually, a continuously-updated `HANDOFF.md`.

**This method worked, but it leaked, and the leaks are instructive.**

- **Ephemeral scratch was treated as durable.** Each session works in a temp directory that does not
  survive it. Session 2's one-off analysis scripts, overlay renders and — most damagingly — the
  original **26-connection hand-traced ground truth** lived only there. The GT was never saved, and
  is permanently lost; every recall figure in the relationship work rests on a 15-edge partial
  reconstruction from a prose summary. Session 3 later rescued what remained by pushing to Hugging
  Face, but by then part of it had already been cleaned up by the OS.
- **Handover documents drift from reality.** Session 3 began by verifying its inherited handover
  against `git status` and found three claims wrong, including one that declared recoverable files
  unrecoverable and another that flagged a completed upload as unverified.
- **Numbers survive in prose but not in code.** Three equipment extents and four instrument
  coordinates that produced the final headline result existed only in a chat transcript. They were
  persisted to `hand_extents/px2368.json` with a reproduction script only at the very end — before
  that, the reported figure could not be regenerated from the repo.
- **Parallel duplication is possible.** The same task was once issued to two sessions; both
  implemented it independently, producing two implementations of one fix and two non-matching reruns
  of the same benchmark. Both were correct; they measured different inputs.

The durable-storage rule that emerged: **Hugging Face Hub is the only shared storage.** Google Drive
was ruled out permanently for a structural reason — `drive.mount()`'s OAuth binds to the
runtime-owning account, so every mount sent a 2FA code to a different person's phone, making
unattended overnight runs impossible.

## 5. Infrastructure, and what it cost

The environment consumed a substantial share of total effort. Recorded honestly because it is the
part most likely to be underestimated next time.

- **GPU:** Colab A100-SXM4-80GB (85.1 GB VRAM). Qwen3-VL 17.5 GB and InternVL3 ~16 GB load
  concurrently.
- **Storage:** HF Hub (`timthy45/pnid-extraction-datasets`, `timthy45/qwen3vl-pnid-domain-base`).
- **A quota incident worth knowing:** every checkpoint push to a fixed path is a new git commit, and
  historical LFS blobs keep counting. **211 commits and 82.5 GB** on one repo silently consumed a
  100 GB free tier. Fixed by deleting and recreating the repo.
- **Xet CDN 403 `SignatureError`** confirmed server-side (persisted with `hf_xet` uninstalled, hit
  public and private repos; `HF_HUB_DISABLE_XET=1` had no effect). Accepted mitigation: hardened
  retry, up to 20 attempts with exponential backoff and jitter. Every real occurrence cleared within
  ≤7 attempts.
- **ModelScope tried as a bypass and rejected:** ~1 Mbps versus ~200 Mbps on HF, a 200× penalty
  (17 GB ≈ 37 h). Also a `ThreadPoolExecutor` timeout could not kill the background download
  thread, whose log lines bled into later cells. Removed from both notebooks with a standing
  "do not re-add."
- **Seven distinct Colab environment failures** were diagnosed and documented during the local
  extraction run alone: a Pillow 12.0.0 regression, torchvision ABI pairing, PaddleOCR segfaulting
  natively at every version tried (worked around by precomputing OCR on the Mac in the pipeline's
  exact render space), HF-hub kernel version mixing, and others.
- **Model-specific pinning:** Molmo2 requires `transformers==4.57.1` while the others run 5.12.1.
- **A compute-cost directive** from management — CPU-only prep locally, GPU sessions doing only
  load/train/infer then `runtime.unassign()` — was **agreed as direction but never implemented in
  code**.

## 6. What the code inventory looks like at close

| Module | Files | Lines | Purpose |
|---|---|---|---|
| `src/relation_bench/` | 39 | ~5,900 | Relationship extraction: line tracing, graph construction, hierarchy, scoring, the v2 entity contract, precision adjudication |
| `src/e2e_bench/` | 29 | ~2,100 | End-to-end harness driving the *real* agent's stage code via Anthropic-shaped client shims |
| `src/extraction_local/` | 14 | ~2,000 | Local entity extraction: Molmo2 pointing, OCR-word pairing, Qwen `call_llm` adapters |
| `src/e2e_harness/` | 12 | ~1,300 | PID2Graph ground truth, graph matching, POC arm runners |
| `src/stage4_symbol_detection/` | 17 | ~1,500 | Stage-4 bake-off: per-candidate load/prompt/parse, eval harness, fine-tune dataset assembly |

Plus 29 notebooks (15 Stage 4, 7 all-stages benchmarking, 7 e2e harness) and 18 markdown documents.

A recurring pattern in the harness code, and a good one: rather than reimplementing the production
agent's logic, the harnesses **drive the real agent code unmodified** through client shims that mimic
`AsyncAnthropic.messages.create()`. That is why the e2e numbers describe the actual pipeline rather
than an approximation of it.
