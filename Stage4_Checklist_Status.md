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
| 2.4 Convert to each model's input format | 🟡 PARTIAL | Candidates locked (2026-07-09): `Qwen/Qwen3-VL-8B-Instruct`, `OpenGVLab/InternVL3-8B-hf`, `allenai/Molmo2-O-7B` (swapped from `Molmo-7B-D-0924` on 2026-07-10 — see 2.5 notes). Common scoring schema defined (`Stage4_Phase2_ModelSetup.ipynb`): `{bbox: [x0,y0,x1,y1], confidence, entity_type}`, tile-local absolute pixel coords. **Qwen3-VL and InternVL3 done** — both load (17.5GB + ~16GB VRAM, comfortably within A100-80GB running concurrently), both round-trip sample images without error using the same `AutoModelForImageTextToText`/`AutoProcessor` pattern. **Molmo2-O-7B done** — loads on the pinned `transformers==4.57.1`, round-trips correctly using the same standard pattern; point-based output (not bbox) handled via a separate parser (`<points coords="frame_id x y"/>`, scaled by 1000). |
| 2.5 Per-model output-format prompt engineering | 🟡 PARTIAL | **Qwen3-VL:** prompt engineered to a compact array-of-arrays format (`[x0,y0,x1,y1,confidence,"type"]`) after two real bugs found and fixed: (1) `repetition_penalty`/`no_repeat_ngram_size` corrupted repeated JSON field names — removed, switched to a key-name-free format instead; (2) unbounded generation could hang indefinitely on dense tiles — added a hard 60s wall-clock `MaxTimeCriteria` (Colab's own Interrupt didn't work on a stuck CUDA call; only a runtime restart recovered it). **Result: 4/10 (40%) parse success** — below the 95% bar, not further pursued. Failure is systematic and density-dependent: every tile with ≥9 ground-truth boxes times out at exactly 60s; every tile with ≤5 boxes succeeds in <11s. Also observed one high-confidence (0.98) fully hallucinated detection on blank canvas — direct evidence for CLAUDE.md rule 6 (never trust recall/confidence alone). **InternVL3:** reused the same prompt/parser as-is (no model-specific tuning needed to get a first result) — **6/10 (60%) parse success**, better than Qwen but the failure pattern is *not* cleanly density-driven (one 2-ground-truth tile still timed out while several 10-16-box tiles succeeded) — something other than symbol count triggers it. Confidence distribution genuinely varies (0.75-0.98, not canned). **Decision: not pursuing further prompt iteration for either model** — when generation completes normally, output is already correctly formatted for both, meaning the prompt successfully conveys the schema; remaining failures look like decoding-level instability (hangs/malformed strings on idiosyncratic content), which prompt wording doesn't reliably fix. This is expected to improve through fine-tuning (the planned next stage), not more zero-shot prompt work. **Molmo candidate swapped** (`Molmo-7B-D-0924` → `Molmo2-O-7B`, 2026-07-10) after 5 escalating environment incompatibilities trying to load the original: (1) `all_tied_weights_keys` attribute missing — known upstream bug `huggingface/transformers#43883`; (2) fixed with wrong type (set vs dict); (3) `tie_weights()` signature mismatch with new kwargs; (4) `stopping_criteria` misplaced (belongs in `generate()` call, not `GenerationConfig`); (5) `super().generate` MRO failure with no clean patch available. Root cause: the 2024-era checkpoint's custom `trust_remote_code` class predates transformers v5's generation-internals restructuring. Pinning an older transformers version was considered and rejected — would require re-switching versions every time the session moves between candidates, an ongoing cost not a one-time fix. Molmo2-O-7B uses the same standard `AutoModelForImageTextToText`/`generate()` pattern as the other two candidates, avoiding this class of bug entirely. Molmo2 requires a pinned `transformers==4.57.1`, different from the 5.12.1 the other two candidates run on — pinned only for the Molmo2 section for now rather than switching versions globally. **Deferred experiment:** Qwen3-VL needed `4.57.0` at its own release; since Molmo2 pins the very next patch, all three candidates might share `4.57.1` — worth testing (pin globally, re-verify Qwen3-VL/InternVL3 still work) next time this notebook is revisited, to potentially eliminate per-candidate version switching entirely. **Molmo2 round-trip + dev-set result: 10/10 (100%) parse success, avg latency 4.40s/tile** — the strongest result of all three candidates, both in parse reliability and speed (vs. Qwen's frequent 60s timeouts and InternVL's slower/inconsistent runs). One notable outlier: `87_640_3840.jpg` predicted 18 detections vs. 7 ground truth — a real over-detection case worth remembering for scoring (same precision-vs-recall caution as CLAUDE.md rule 6), but not alarming on its own. Plausible explanation for the strong result: Molmo2 is natively built for point-based localization, so this task doesn't fight its trained output format the way bbox-JSON does for the other two. `confidence`/`entity_type` are `None` for every Molmo2 detection (format has no native fields for either) — by design, not a bug; typing/confidence-based scoring will need to exclude this candidate or handle it specially. |
| 2.6 Session-budget gate | ✅ DONE | Real tile counts computed on the 20 frozen test sheets via `compute_tile_grid` (2.3): 127 tiles total, avg 6.3/sheet (range 1-20 depending on sheet size, e.g. `194`/`196` at 3800×2458 need 20 tiles each). Extrapolated from dev-set latencies (2.5): Qwen3-VL 1.34h, InternVL3 0.97h, Molmo2-O-7B 0.16h — all three comfortably fit a single ~12h Colab session with 10+ hours of margin each. No batching/checkpointing needed. |

---

## Phase 3 — Metric Harness

| Item | Status | What exists / what's left |
|---|---|---|
| 3.1 Two scoring functions (detection mAP/F1 + typing acc) | ✅ DONE | Implemented in `src/stage4_symbol_detection/scoring.py`: `precision_recall_f1` (class-agnostic detection, Gupta), `average_precision` (confidence-ranked, returns `None` for Molmo2 since it has no native confidence — by design, not a bug), `typing_accuracy_gt_crop` (+ per-class + rare-class recall, <20 instances). **Part B scope decision:** typing scored on GT-cropped symbols (localization handed to model for free) — isolates pure typing ability. All confirm-bar cases tested and passing (`test_scoring.py`): perfect→1.0, empty→0.0, half-correct→sane middle, for both detection and typing. |
| 3.2 Unified match metric fair to boxes AND points | ✅ DONE | **Decision: point-in-GT-box hit.** Box predictions use their center point; point predictions (Molmo2) use the point directly — no artificial box-size invented for Molmo, no IoU needed for a model that never predicted a box. Same `match_predictions_to_gt` function called for all three candidates, no model-specific branch. Tested: point clearly inside → hit, clearly outside → miss, boundary inclusive; one prediction can't double-match overlapping GT boxes. |
| 3.3 Tile→sheet stitch + NMS dedup | ✅ DONE | Implemented in `src/stage4_symbol_detection/stitch.py`: `remap_tile_to_sheet` (matches the real agent's `tile_to_drawing` convention — add tile origin, `Agent_Pipeline_Facts.md` §1), `stitch_and_dedup` (proximity-based clustering per entity type — fair to points and boxes alike, since IoU isn't defined for a point). Tested: a boundary symbol detected in 2 overlapping tiles appears exactly once after dedup (keeping the higher-confidence copy), stitched count ≤ raw sum, and genuinely distinct far-apart symbols are never merged. |
| 3.4 Optional incumbent (Claude) reference column | ⛔ DEFERRED | Explicitly deferred by decision (2026-07-10) — not the pass bar per the checklist itself, and requires live cloud agent access not currently prioritized. Revisit later. |
| 3.5 IoU/threshold constants | ✅ DONE | Fixed in `scoring.py`: `IOU_THRESHOLD = 0.5` (documented for box-format analyses, though not the primary match rule per 3.2), `BOX_FORMAT = "xyxy_absolute"`, `RARE_CLASS_MAX_INSTANCES = 20`. Single source of truth, imported everywhere rather than scattered magic numbers. |
| 3.6 Set tolerance / pass bar | 🟡 PARTIAL | **Provisional pass bar set** (decision 2026-07-10): detection (Gupta) mAP@0.5 ≥ 0.70 OR F1 ≥ 0.70; typing (Kaggle, GT-crop mode) accuracy ≥ 0.80. Explicitly flagged as provisional, not derived from any real reference point yet. **Fallback-detector trigger genuinely blocked, not just deferred**: the spec ties it to "`[X]`% of Claude's Stage-4 recall, threshold to be set from the baseline run" (`PID_Local_Substitution_Spec.md` §1) — that baseline run is exactly what 3.4 defers, so this number cannot be set honestly without it. Revisit both together when the incumbent run happens. |

---

## Phase 4 — Zero-Shot Baseline

| Item | Status | What exists / what's left |
|---|---|---|
| 4.1 Qwen3-VL zero-shot | 🔴 TODO (deferred) | **Not the same as the earlier 10-sample Kaggle dev-set parse-rate check** — that ran on the wrong dataset (Kaggle, synthetic/typing-only) and measured parse success, not actual detection recall via the Phase 3 harness. Real run (on the 20 frozen Gupta test sheets) deliberately deferred (2026-07-10) — decided not worth doing yet. |
| 4.2 InternVL3 zero-shot | 🔴 TODO (deferred) | Same as 4.1 — deferred, not done. |
| 4.3 Molmo2 zero-shot | 🔴 TODO (deferred) | Same as 4.1 — deferred, not done. Cheapest/most promising candidate to actually run first when this phase is picked back up (100% parse rate, ~0.16h budget per 2.6), but not yet run through the real harness on Gupta. |
| 4.4 Compare with validity gate | 🔴 TODO (deferred) | Blocked on 4.1-4.3. |

---

## Phase 5 — Fine-Tuning

**⚠️ Decision (2026-07-10): fine-tune all 3 candidates, not just the winner/top 2.** Deviates
from the checklist's own scoping ("Fine-Tuning (winner, or top 2)") — explicit user decision,
expense not a constraint. Phase 4 (zero-shot baseline) is still deferred; this decision means
Phase 4's original *purpose* (narrowing candidates before expensive fine-tuning) no longer
gates Phase 5 — all three proceed regardless of what a zero-shot bake-off would have shown.
Phase 4 may still be worth doing later for its own sake (a real zero-shot number is useful
context even if it's not gating anything), but it's not a blocker for Phase 5 anymore.

| Item | Status | What exists / what's left |
|---|---|---|
| 5.1 Build domain-adaptation training set | ✅ DONE | Implemented in `src/stage4_symbol_detection/finetune_dataset.py`: `assemble_manifest()` collects Gupta's 72 train sheets + all Kaggle pretrain images into a model-agnostic manifest, asserting zero train/test leak against `test_ids.json` before returning anything (not optional — no code path skips it). Tested against a synthetic fake repo with a deliberately planted leak, confirmed `AssertionError` is raised (not silently passed) and names the offending sheet ID. **Per-model training-format conversion also done** (`training_format.py`): converts ground-truth YOLO labels into each candidate's own inference-output convention — box-JSON array-of-arrays for Qwen3-VL/InternVL3, `<points>` tags for Molmo2 — so training targets are guaranteed parseable by the exact same parser used for scoring (round-trip tested, not assumed). Gupta boxes get generic `entity_type: "symbol"` (class-agnostic per rule 5); Kaggle boxes get real class names from `classes.json` when available. Caught and fixed one real edge-case bug via testing: empty-target formatting produced a `coords=""` tag the parser's own regex could never match. |
| 5.2 Task-neutral domain fine-tune | ⛔ RESEQUENCED | **Decision (2026-07-10):** doing combined detection fine-tune (5.3-style, directly on the raw base) FIRST for Stage 4's own deliverable, deferring the proper task-neutral domain-adapt pass to end-of-Stage-4. Rationale: separate layering (domain-adapt → clean base → detection LoRA) roughly doubles training wall-clock vs. one combined pass (~17-25h vs ~8.5-12h per model, rough estimate, not measured), and Stage 4 doesn't need the domain base to be reusable — only later stages do. **Reconciled with CLAUDE.md rule 3, not violated:** the deferred domain-adapt pass will branch fresh from the same untouched original weights, producing an independent checkpoint for future-stage reuse — NOT stacked on top of Stage 4's detection-tuned checkpoint. Two separate checkpoints from one base, not one polluted checkpoint. |
| 5.3 Stage-4 detection LoRA adapter | 🔴 TODO | Effectively merged with 5.2 per the 2026-07-10 resequencing decision — training a single combined LoRA directly on the raw base for the detection task (using `training_format.py` targets), not a separate adapter on top of a pre-existing domain base. Not yet built. |
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
