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
| 0.1 Provision Colab Pro+ GPU | ✅ DONE | GPU runtime confirmed: **NVIDIA A100-SXM4-80GB**, 85.1 GB VRAM (`torch.cuda` sees it). Exceeds the ideal 40GB bar — no QLoRA/offload needed for 8B fine-tuning. |
| 0.2 Mount Drive for persistence | ✅ DONE | Drive mounts; `MyDrive/pid_project/data/` exists and persists. (Checklist names it `pid_stage4/` — cosmetic difference, reuse `pid_project`.) |
| 0.3 Install dependencies | ✅ DONE | torch, transformers, accelerate, vllm, pycocotools, supervision, kagglehub, kaggle, qwen_vl_utils, einops, timm all import cleanly on the GPU runtime. `requirements.txt` pinned to Drive (`pid_project/data/requirements.txt`). vllm needed a CUDA 13 runtime fix (torch is cu128; preloaded `libcudart.so.13` via `ctypes` since `LD_LIBRARY_PATH` set at runtime didn't reach vllm's `dlopen`). |
| 0.4 Set up MLflow tracking | ⛔ N/A | Superseded — tracking via `results.csv` + `experiments/stage4/v*.md` per CLAUDE.md rule #8. |

---

## Phase 1 — Dataset Acquisition & Integrity

| Item | Status | What exists / what's left |
|---|---|---|
| 1.1 Download Gupta | ✅ DONE | Downloaded to Drive via Colab wget (direct from Zenodo). |
| 1.2 Extract Gupta | ✅ DONE | Sheet-level count confirmed at `PID_Dataset/0__raw_data/sheets/{train,test}`: 72 train + 20 test = 92, `assert`ed directly. (Tile-level 642/215 split from Step 1 is separate/downstream — sheets are the ground truth for this check.) |
| 1.3 Download Kaggle | ✅ DONE | **Re-downloaded and replaced** — original Drive upload was broken/incomplete (only 6,591/30,000 labels). Fresh download via `kaggle datasets download -d hristohristov21/pid-symbols` verified complete (30,000/30,000 labeled, 195,759 instances — matches dataset card) before swapping in and deleting the broken copy. |
| 1.4 Extract Kaggle | ✅ DONE | Confirmed against dataset card: 32 classes, 30,000 images, 195,759 instances, all images labeled. Supersedes the earlier undercounted 43,055-instance figure (that was measured on the broken copy). |
| 1.5 Verify annotation integrity | ✅ DONE | Gupta: 0 orphans, 0 unannotated across train (72) + test (20). Kaggle: after the re-download fix, 30,000/30,000 images labeled — 0 orphans, 0 unannotated. (Root cause of the original gap: broken/incomplete Drive upload, not the Kaggle source itself — confirmed by re-download matching the dataset card exactly.) |
| 1.6 Visual spot-check | ✅ DONE | 5 Kaggle + 5 Gupta overlays rendered, displayed inline, and saved to `Drive/overlays/`. Boxes land correctly on symbols in both datasets — coordinate parsing (normalized cxcywh → pixels) confirmed correct. Kaggle class IDs vary appropriately across tiles (e.g. 18/29/17/30 on one tile, tight boxes on distinct symbol types); one legend/key tile showed all-class-0 but is a one-off outlier tile, not a systemic labeling bug. Gupta boxes densely and correctly cover a real vessel drawing (V-001), all class 0 as expected for class-agnostic labeling. |

**Bonus already done (not in checklist but valuable):** image-health check — 0 unreadable, Kaggle uniform 1280×1280, Gupta 224–1000 range. Feeds tiling/resize decisions.

---

## Phase 2 — Data Preparation

| Item | Status | What exists / what's left |
|---|---|---|
| 2.1 Lock the test split | ✅ DONE | Frozen `test_ids.json` written to Drive (`Stage4_Phase2_Data_Preparation.ipynb`), sheet-level IDs for 72 train / 20 test, zero overlap asserted. Physical separation already existed via Gupta's own `sheets/{train,test}` folder structure. The cell hard-fails on re-run if the split ever differs from what's frozen, per CLAUDE.md hard rule #1. **Still open (belongs to 2.3, not 2.1):** reconciling this sheet-level split with the tile-level 642/215 split once agent tiling scheme is pulled — must confirm tiles don't leak across the sheet boundary once Gupta gets re-tiled to match Stage 3's scheme. |
| 2.2 Fix the two-part metric | ✅ DONE | **Decision established:** Gupta = class-agnostic (detection only), Kaggle = 32 typed classes (typing only). **`classes.json` complete**: class 0 = `Not_used` (0 instances confirmed), classes 1-32 named + mapped to a 6-category reference ontology (`valve`/`safety_device`/`pipeline`/`measurement`/`nozzle`/`asset` — illustrative examples from agent code, explicitly NOT an authoritative tenant ontology, since none exists per `Agent_Pipeline_Facts.md` §3). **Coverage reported both ways** (category-count hides skew): 6/6 categories (100%) represented, but instance-weighted: valve 48.2%, pipeline 17.4%, measurement 17.0%, nozzle 9.1%, asset 5.4%, safety_device 2.9%. This skew — not the flat 100% — is what must sit next to the typing score, per CLAUDE.md rule 5. **Still open:** the two scoring paths (detection harness on Gupta, typing harness on Kaggle) are designed but not yet built as code. |
| 2.3 Tile to match agent Stage 3 | ✅ DONE | Implemented in `Stage4_Phase2_Data_Preparation.ipynb`: exact 1024×1024/205px-overlap/819px-stride uniform grid, edge tiles clamped not repositioned. Verified on sample sheet `132.jpg` (604×1044): 2 tiles (both edge, since sheet < 1 tile wide), 5 original boxes → 7 per-tile boxes (2 boundary boxes correctly duplicated across the tile seam), zero drops asserted. Visual overlay confirmed boxes land correctly on symbols in tile-local coords. **Known simplification:** no title-block carve-out (would need Stage 2's title-block detector, which this repo doesn't have) — tiles the full sheet instead, which mirrors the real agent's own `full_page`/`title_block_rejected` fallback path, not an invented shortcut. |
| 2.4 Convert to each model's input format | 🟡 PARTIAL | Candidates locked (2026-07-09): `Qwen/Qwen3-VL-8B-Instruct`, `OpenGVLab/InternVL3-8B-hf`, `allenai/Molmo-7B-D-0924`. Common scoring schema defined (`Stage4_Phase2_ModelSetup.ipynb`): `{bbox: [x0,y0,x1,y1], confidence, entity_type}`, tile-local absolute pixel coords. **Qwen3-VL and InternVL3 done** — both load (17.5GB + ~16GB VRAM, comfortably within A100-80GB running concurrently), both round-trip sample images without error using the same `AutoModelForImageTextToText`/`AutoProcessor` pattern. Molmo not yet started (expected to differ — native point-based output, not bbox). |
| 2.5 Per-model output-format prompt engineering | 🟡 PARTIAL | **Qwen3-VL:** prompt engineered to a compact array-of-arrays format (`[x0,y0,x1,y1,confidence,"type"]`) after two real bugs found and fixed: (1) `repetition_penalty`/`no_repeat_ngram_size` corrupted repeated JSON field names — removed, switched to a key-name-free format instead; (2) unbounded generation could hang indefinitely on dense tiles — added a hard 60s wall-clock `MaxTimeCriteria` (Colab's own Interrupt didn't work on a stuck CUDA call; only a runtime restart recovered it). **Result: 4/10 (40%) parse success** — below the 95% bar, not further pursued. Failure is systematic and density-dependent: every tile with ≥9 ground-truth boxes times out at exactly 60s; every tile with ≤5 boxes succeeds in <11s. Also observed one high-confidence (0.98) fully hallucinated detection on blank canvas — direct evidence for CLAUDE.md rule 6 (never trust recall/confidence alone). **InternVL3:** reused the same prompt/parser as-is (no model-specific tuning needed to get a first result) — **6/10 (60%) parse success**, better than Qwen but the failure pattern is *not* cleanly density-driven (one 2-ground-truth tile still timed out while several 10-16-box tiles succeeded) — something other than symbol count triggers it. Confidence distribution genuinely varies (0.75-0.98, not canned). **Decision: not pursuing further prompt iteration for either model** — when generation completes normally, output is already correctly formatted for both, meaning the prompt successfully conveys the schema; remaining failures look like decoding-level instability (hangs/malformed strings on idiosyncratic content), which prompt wording doesn't reliably fix. This is expected to improve through fine-tuning (the planned next stage), not more zero-shot prompt work. Molmo not yet started — will need a different approach (point-based, not bbox). |
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
