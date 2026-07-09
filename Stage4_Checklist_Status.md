# Stage 4 — Symbol Detection Benchmarking · Status Audit

**Purpose:** the uploaded Stage 4 checklist, annotated with what is ALREADY DONE (and reusable) from the prior block-based work, versus what still needs doing under the substitution plan.

**Status legend:**
- ✅ **DONE** — completed and reusable as-is
- 🟡 **PARTIAL** — some work done, needs adaptation to the checklist's bar
- 🔴 **TODO** — not started
- ⛔ **N/A** — made irrelevant by the substitution reframe

---

## Phase 0 — Environment Setup

| Item | Status | What exists / what's left |
|---|---|---|
| 0.1 Provision Colab Pro+ GPU | 🟡 PARTIAL | Colab Pro+ available and used, but Step 1 ran on **CPU**. Need a GPU runtime + record `nvidia-smi` (model + VRAM) from Phase 4 onward. |
| 0.2 Mount Drive for persistence | ✅ DONE | Drive mounts; `MyDrive/pid_project/data/` exists and persists. (Checklist names it `pid_stage4/` — cosmetic difference, reuse `pid_project`.) |
| 0.3 Install dependencies | 🟡 PARTIAL | Inspection libs (PIL/numpy/pandas) installed. **Still need:** torch, transformers, vllm/lmdeploy, mlflow, pycocotools, supervision, kagglehub, model loaders. No pinned `requirements.txt` yet. |
| 0.4 Set up MLflow tracking | 🔴 TODO | Not done. Was queued as task T-008; now mandatory and blocking (everything logs to it). Point at Drive-backed store, experiment `pid-stage4`. |

---

## Phase 1 — Dataset Acquisition & Integrity

| Item | Status | What exists / what's left |
|---|---|---|
| 1.1 Download Gupta | ✅ DONE | Downloaded to Drive via Colab wget (direct from Zenodo). |
| 1.2 Extract Gupta | 🟡 PARTIAL | Extracted; counted 4,590 images and 642/215 **tiles**. **NOT confirmed to checklist bar:** need `assert n_annotated_sheets == 92` (72 train + 20 test at the **sheet** level). Tile counts ≠ sheet counts — the 92-sheet verification is still open. |
| 1.3 Download Kaggle | ✅ DONE | Downloaded, extracted locally, zipped, uploaded to Drive. |
| 1.4 Extract Kaggle | ✅ DONE | Confirmed 32 classes, ~30k images, 43,055 instances parsed, all 32 classes present, balanced (rarest 1,120). Folder de-duplicated (stray `archive/` removed), train.txt + val.txt in place. |
| 1.5 Verify annotation integrity | 🔴 TODO | Not done. Need: zero orphan annotations, zero unannotated images in labeled splits. `assert orphans == 0 and unannotated == 0`. (Note: earlier saw 6,591 labels vs 30k images on Kaggle — must confirm whether unlabeled = intentional negatives or a mismatch.) |
| 1.6 Visual spot-check | 🔴 TODO | Not done. Render 5 Gupta + 5 Kaggle with boxes overlaid; confirm boxes land on symbols (catches xywh/xyxy + normalized/absolute bugs). Save 10 overlays to Drive. |

**Bonus already done (not in checklist but valuable):** image-health check — 0 unreadable, Kaggle uniform 1280×1280, Gupta 224–1000 range. Feeds tiling/resize decisions.

---

## Phase 2 — Data Preparation

| Item | Status | What exists / what's left |
|---|---|---|
| 2.1 Lock the test split | 🟡 PARTIAL | Gupta ships 642/215 tile split; the **sheet-safe split concern** was already identified (tiles from one sheet must not leak across splits) — this is exactly checklist 2.1. **Still need:** frozen `test_ids.json`, assert zero train/test overlap, assert 72/20 at sheet level. The Step-2 inspection to map tiles→sheets was planned but not run. |
| 2.2 Fix the two-part metric | 🟡 PARTIAL | **The core insight is already established:** Gupta = class-agnostic (detection only), Kaggle = 32 typed classes (typing only). This was independently discovered in Step 1 and IS the crux this item describes. **Still need:** build the two scoring paths + `classes.json` mapping Kaggle's 32 → agent ontology, and record ontology-coverage %. |
| 2.3 Tile to match agent Stage 3 | 🔴 TODO | Blocked on agent code. Need the agent's exact 1024×1024 overlapping-tile scheme + title-block fence, then remap annotations. (Gupta arrives pre-tiled at varying sizes — must reconcile with agent's 1024 tiling.) |
| 2.4 Convert to each model's input format | 🔴 TODO | Blocked on base-VLM candidates. Not started. |
| 2.5 Per-model output-format prompt engineering | 🔴 TODO | Blocked on candidates + agent Stage 4 output schema. Not started. This is flagged as "most of the battle." |
| 2.6 Session-budget gate | 🔴 TODO | Not started. Time 5 tiles, extrapolate to full 20-sheet pass, confirm fits in 12h. |

---

## Phase 3 — Metric Harness

| Item | Status | What exists / what's left |
|---|---|---|
| 3.1 Two scoring functions (detection mAP/F1 + typing acc) | 🟡 PARTIAL | The **B2 OCR scoring harness** (CER/exact-match) is a reusable *pattern* but wrong metrics for Stage 4. Detection mAP@0.5/F1 on Gupta + typing accuracy + rare-class recall on Kaggle = new build. The perfect/empty/half sanity-check discipline carries over. |
| 3.2 Unified match metric fair to boxes AND points | 🔴 TODO | New. Critical because Molmo emits points, Qwen/InternVL emit boxes. Point-in-GT-box rule. Not started. |
| 3.3 Tile→sheet stitch + NMS dedup | 🔴 TODO | New. Mirror agent's cross-tile NMS before scoring. Not started. |
| 3.4 Optional incumbent (Claude) reference column | 🔴 TODO | Optional. Do while cloud access live. Not started. |
| 3.5 IoU/threshold constants | 🔴 TODO | Fix IoU 0.5, box xyxy absolute, point-in-box rule. Not started. |
| 3.6 Set tolerance / pass bar | 🔴 TODO | Define pass bar as **mAP@0.5 / F1 (NOT recall alone)** + the fallback-detector trigger threshold. Write down before running. Not started. |

---

## Phase 4 — Zero-Shot Baseline

| Item | Status | What exists / what's left |
|---|---|---|
| 4.1 Qwen3-VL zero-shot | 🔴 TODO | New — replaces old YOLO/RT-DETR plan. |
| 4.2 InternVL3 zero-shot | 🔴 TODO | New. |
| 4.3 Molmo zero-shot | 🔴 TODO | New. |
| 4.4 Compare with validity gate | 🔴 TODO | New. Check parse-failure rates first; high failure → ranking is noise. |

---

## Phase 5 — Fine-Tuning

| Item | Status | What exists / what's left |
|---|---|---|
| 5.1 Build domain-adaptation training set | 🟡 PARTIAL | Datasets ready; the **train-only, zero-test-leak discipline** already understood. Still need the assembled FT set in the winner's format + `intersection(train,test)==empty` assert. |
| 5.2 Task-neutral domain fine-tune | 🔴 TODO | New (QLoRA). The general LoRA/QLoRA-on-Colab plan from earlier docs carries over as method. |
| 5.3 Stage-4 detection LoRA adapter | 🔴 TODO | New — adapter-on-top-of-base layering. |
| 5.4 Run fine-tuned on test set | 🔴 TODO | New. |

---

## Phase 6 — Selection & Decision

| Item | Status |
|---|---|
| 6.1 Final comparison | 🔴 TODO |
| 6.2 Apply pass bar | 🔴 TODO |
| 6.3 Lock base model | 🔴 TODO |
| 6.4 Reproducibility check (greedy, fixed seed) | 🔴 TODO |

---

## Phase 7 — Handoff

| Item | Status |
|---|---|
| 7.1 Document base for reuse | 🔴 TODO |
| 7.2 Log metric limitation (real typing never tested) | 🟡 PARTIAL — the limitation itself (synthetic typing, 73%→27% risk) is already understood from Step 1; just needs formal write-up when reached. |

---

## Summary

**Done & reusable (✅):** Drive mount, Gupta download, Kaggle download+extract+clean. Plus bonus image-health check.

**Partial — real work banked, needs finishing to the checklist bar (🟡):**
- Colab GPU (used CPU so far)
- Dependencies (inspection libs only)
- Gupta extract (need 92-sheet assert)
- Test split (concern identified, not frozen)
- Two-part metric (insight established, not built)
- Scoring-harness pattern (from B2, wrong metrics)
- Train-leak discipline (understood, not asserted)
- Metric limitation note (understood, not written)

**Not started (🔴):** MLflow, annotation integrity, visual spot-check, all of agent-matched tiling, output-format prompt work, the full metric harness (3.2–3.6), all zero-shot runs, all fine-tuning, selection, handoff.

**Effective position:** solid through **Phase 1 (~70%)**, with the conceptual groundwork for Phase 2 already laid. The expensive data work is banked. What remains is mostly net-new and correctly sequenced by the checklist.

## Immediate next 3 (in order)
1. **MLflow (0.4)** + full deps (0.3) — unblocks all logging.
2. **Close Phase 1 confirms** — 92-sheet assert (1.2), integrity (1.5), visual overlay (1.6).
3. **Pull agent code** (Stage 3 tiling + Stage 4 output schema) — unblocks Phase 2.3 / 2.5.
