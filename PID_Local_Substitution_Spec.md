# P&ID Intelligence Agent — Local Model Substitution Spec

**Version:** 5.0
**Status:** Draft for review
**Owner:** _[your name]_
**Last updated:** 2026-07-09

---

## 1. Goal

The P&ID intelligence agent already exists and works. It relies on **cloud models** at several stages: `claude-sonnet-4-6`, `claude-opus-4-8`, and the Google Cloud Vision API.

This project has exactly one objective: **replace every cloud LLM / VLM / API call in the agent with the best local counterpart**, fine-tuned per stage, then benchmark the local version against the original to confirm it produces good results.

**This is a substitution project, not a redesign.**
- The agent's architecture, stages, and order are **fixed**. We do not add, remove, reorder, or re-architect stages.
- **No ML models** (no YOLO, RT-DETR, Relationformer, U-Net, Siamese, etc.). The substitutes are local **VLMs, LLMs, and OCR models** only.
- Stages that are **already offline** (pure Python / OpenCV / NetworkX / regex) are **left untouched** — out of scope.

**One documented exception (Plan B for the highest-risk stage):** the no-ML rule holds by default, but if the fine-tuned VLM at Stage 4 cannot reach **[X]% of Claude's Stage-4 recall** (threshold to be set from the baseline run), a dedicated detector re-enters scope for that stage only. This is the single stage where VLMs are structurally weakest (exhaustive detection on dense tiles), so it gets an explicit off-ramp rather than no fallback.

---

## 2. Scope Rule

For each agent stage, ask one question: **does it call a cloud model or API?**

- **Yes** → in scope. Substitute with a local VLM / LLM / OCR model.
- **No** (already local Python / OpenCV / NetworkX / regex) → out of scope. Do not touch.

That is the entire decision.

---

## 3. All Stages — Substitution Table

Every stage in the agent. Colour = benchmark tier. **In scope** = uses a cloud model today.

**Tier legend:** 🟥 FULL · 🟧 BAKE-OFF · 🟩 DEFAULT + smoke · ⬜ UNIT-TEST (offline, out of scope)

| Stage | What it does | Original model | Scope | Local replacement | Tier |
|---|---|---|---|---|---|
| **Stage 0 — Ingestion** | Render PDF/image → normalised PNG, build DrawingDocument spine | None (PyMuPDF + OpenCV) | ⬜ Out | — no change | ⬜ Unit-test |
| **Stage 1 — Sheet Classification** | Label each page: pid_drawing / legend / notes / cover | `claude-sonnet-4-6` | 🟩 **In** | Shared base VLM (reused; 2B fallback if VRAM-tight) | 🟩 Default |
| **Stage 1.5 — Page OCR** | Read every word + pixel bbox — single cached call | Google Cloud Vision | 🟧 **In** | PaddleOCR PP-OCRv5 · docTR · Qwen3-VL-OCR — *judged on downstream tag-association, not isolated CER* | 🟧 Bake-off |
| **Notes Extraction** | Parse numbered notes from OCR text into notes_index | None (pure regex) | ⬜ Out | — no change (LLM fallback = LAST priority) | ⬜ Unit-test |
| **Stage 2 — Title Block** | Extract drawing metadata (number, revision, title, site) | `claude-sonnet-4-6` | 🟩 **In** | Shared base VLM (reused) | 🟩 Default |
| **Stage 3 — Tile Segmentation** | Cut page into overlapping 1024×1024 tiles; fence off title block | None (pure Python / OpenCV) | ⬜ Out | — no change | ⬜ Unit-test |
| **Stage 5 — OCR Ensemble** | Triple-call Vision per tile + Sonnet tiebreaker on low-conf words | `claude-sonnet-4-6` + Vision *(disabled)* | 🟩 **In** | PaddleOCR multi-pass + shared base VLM tiebreaker | 🟩 Default |
| **Stage 4 — Symbol Detection** | Find every symbol per tile; read tag; classify entity type | `claude-sonnet-4-6` | 🟥 **In** | Shared base + **Stage-4 detection LoRA**; base chosen here on Gupta ground truth | 🟥 Full |
| **Stage 6 — Line Tracing** | Trace pipe/signal runs; record endpoints + line type | None (OpenCV Hough + skeletonize) | ⬜ Out | — no change | ⬜ Unit-test |
| **Stage 10 — Ontology Mapping** | Validate entity types against tenant ontology | None (inlined into Stage 11, deterministic) | ⬜ Out | — no change | ⬜ Unit-test |
| **Stage 10.5 — Skid Grouping** | Assign components to equipment skids via ROI crops | `claude-sonnet-4-6` | 🟧 **In** | Shared base VLM (reused, + LoRA if needed) | 🟧 Bake-off |
| **Stage 11 — Graph Construction** | Fuse detections + line graph + ontology → BundleEntity/BundleRelation | None (deterministic Python / NetworkX) | ⬜ Out | — no change | ⬜ Unit-test |
| **Dynamic Cropping — Low-conf refinement** | Zoom into uncertain entities, reassess in place | `claude-opus-4-8` *(disabled)* | 🟩 **In** | Shared base VLM (reused) | 🟩 Default |
| **Stage 13 — Entity Validation** | Verify every entity against drawing; keep / correct / remove | `claude-sonnet-4-6` | 🟧 **In** | Rules-first → shared base VLM adjudicator | 🟧 Bake-off |
| **Stage 12 — Relation Validation** | Confirm every connection against drawing; populate annotations | `claude-sonnet-4-6` *(disabled)* | 🟧 **In** | Rules-first → shared base VLM adjudicator | 🟧 Bake-off |
| **Finalize** | Pick best bundle (12→13→11), persist rive_ontology.json | None (deterministic) | ⬜ Out | — no change | ⬜ Unit-test |

---

## 4. Summary

**16 stages total.**

- **9 in scope** (use a cloud model): Stage 1, 1.5, 2, 5, 4, 10.5, dynamic cropping, 13, 12.
- **7 out of scope** (already offline): Stage 0, notes extraction, Stage 3, 6, 10, 11, finalize.

**Tier distribution across the 9 in-scope stages:**

| Tier | Stages | Why |
|---|---|---|
| 🟥 FULL (1) | Stage 4 Symbol Detection | Hardest task, real candidate choice, highest impact on output |
| 🟧 BAKE-OFF (4) | Stage 1.5 OCR · 10.5 Skid Grouping · 13 Entity Validation · 12 Relation Validation | Real choice between local candidates/configs; quality materially affects output |
| 🟩 DEFAULT + smoke (4) | Stage 1 Sheet Class. · 2 Title Block · 5 OCR Ensemble · Dynamic Cropping | Obvious single answer or disabled-by-default; drop in and smoke-test |

> **Every in-scope stage is benchmarked against the cloud agent's own output.** The tier does not decide *whether* to benchmark — it decides *how much work the swap needs*. FULL (Stage 4) = model bake-off + fine-tune + iterate; this also picks the shared base VLM. BAKE-OFF (10.5, 13, 12, plus OCR at 1.5) = for VLM stages, compare only *prompts/config* on the already-chosen shared base (not new models); for OCR, compare engines. DEFAULT = reuse the shared base with a prompt, smoke-test it matches the cloud output.

---

## 5. Model Strategy — One Shared Base VLM, Reused Across Stages

**Core principle: fine-tune once, reuse everywhere. Do not fine-tune a separate model per stage.**

The VLM stages (4, 10.5, 13, 12, dynamic cropping) are not different *tasks* needing different *weights* — they are the same underlying skill (understand a P&ID region, answer a structured question about it) with different **prompts**. Training five separate checkpoints would multiply training cost, serving VRAM, and maintenance for no benefit.

**The strategy:**

1. **Fine-tune the domain base ONCE (task-neutral).** Take one base VLM and fine-tune it on P&ID data purely for **domain adaptation** — what P&ID symbols look like, tag formats, drawing conventions. This is deliberately **not** trained on a detection objective. It is the task-neutral shared base that every VLM stage inherits.
2. **Stage 4 detection is a LoRA adapter ON TOP of the base — not baked in.** Teaching the model to output a box for every symbol is a specific output mode. It lives in a **detection LoRA adapter** loaded only at Stage 4. This keeps the shared base clean so the reasoning stages (10.5, 13, 12) are not skewed toward detection output.
3. **Reuse the clean base across reasoning stages via prompts.** Stage 10.5 (grouping), 13/12 (validation), dynamic cropping, Stage 2 (title block), Stage 1 (gate) all call the **same domain base** with **stage-specific prompts** — no detection adapter, no new weights.
4. **Add further LoRA adapters only if measured.** If a specific reasoning stage underperforms on the clean base, add a small stage-specific adapter for it. Do not maintain separate full models.

**Fine-tuning budget: 1 task-neutral domain fine-tune + 1 Stage-4 detection LoRA + at most a small number of stage adapters where measured. Not 5 separate models.**

> The layering matters: **domain adaptation = shared base; detection = adapter on top.** Training "at Stage 4" with a detection objective would bake detection bias into the base and skew every downstream reasoning stage. Train the neutral base first, add the detection adapter second.

### VLM candidates — base selection bake-off

The base VLM is chosen **once**, from a three-way bake-off at **Stage 4**, scored against **Gupta symbol-detection ground truth**. Stage 4 is chosen as the selection stage because it has full real ground truth, it is the hardest and highest-leverage task, and its winner becomes the shared base every other VLM stage reuses.

**Caveat carried forward:** the base is picked on detection, but it must also serve the reasoning stages (10.5, 13, 12). Molmo's edge is pixel-pointing (a localization skill) — strong for Stage 4 but not necessarily for validation reasoning. So: if the Stage-4 winner later underperforms on a reasoning stage's own ground-truth benchmark, that stage gets a LoRA adapter (or, in the worst case, a different base is reconsidered). Track this rather than assuming the detection winner is automatically best everywhere.

| Candidate | Why in contention | Licence |
|---|---|---|
| **Qwen3-VL (7B/8B)** | Strongest general open VLM in class; best object-localization + OCR heritage; fits a single 24 GB GPU | Qwen licence |
| **InternVL3 (8B)** | Strongest MIT-licensed line; genuine Qwen competitor on document/detection; cleaner licence | MIT |
| **Molmo (7B)** | Native **pointing** — grounds answers to pixel locations; strong for Stage 4 localization, but verify it holds up on the reasoning stages | Apache-2.0 (open data + code) |

> The base is selected on Stage 4 ground-truth accuracy. The reasoning stages reuse it and are each benchmarked on their own ground truth; a stage that underperforms gets an adapter.

### Full local model stack

| Role | Local model | Used at |
|---|---|---|
| Cheap gate | Shared base VLM (reused with a gate prompt); Qwen3-VL-2B only as a VRAM fallback | Stage 1 |
| **Shared base VLM** | **Stage 4 bake-off winner** (Qwen3-VL-8B / InternVL3-8B / Molmo-7B), domain fine-tuned once | Stages 4, 5 (tiebreaker), 10.5, 13, 12, dynamic cropping |
| OCR | PaddleOCR PP-OCRv5 (FT recognizer); docTR / Qwen3-VL-OCR as bake-off candidates | Stages 1.5, 5 |
| Small LLM (structured text) | Shared base VLM (reused) — no separate model | Stage 2 |

---

## 6. Datasets (fine-tuning + evaluation only)

| Dataset | Use |
|---|---|
| **Mohit Gupta PID_Dataset** · `zenodo.org/records/8028570` | FT + eval Stage 4 (real sheets); OCR tag crops for Stage 1.5 / 5 |
| **PID2Graph OPEN100** · `zenodo.org/records/14803338` | Eval grouping / validation stages against real graph ground truth |
| **Kaggle P&ID Symbols** · `kaggle.com/datasets/hristohristov21/pid-symbols` | Extra symbol examples for Stage 4 fine-tuning |

Real data is the only valid judge. Synthetic (if used) is fine-tuning volume only, never in a test set.

---

## 7. Method — Isolated, Ground-Truth Benchmarking

**Each stage is benchmarked in complete isolation, scored against dataset ground truth.**

We are not chaining stages during benchmarking, and we are not scoring against the cloud agent's output. For each stage: construct its input, run the local model, score the output against the **real dataset ground truth**, judge the accuracy. This measures "how good is this local stage at the actual task," which is a stronger and more objective bar than matching a nondeterministic cloud model.

**Base-model selection (do once, at Stage 4):**
```
A. Bake off the base VLM candidates (Qwen3-VL vs InternVL3 vs Molmo) on
   Stage 4 symbol detection, scored against Gupta ground-truth boxes.
   Stage 4 has full real ground truth and picks the base everything reuses.
B. Domain fine-tune the winner ONCE, task-neutral. This is the shared base.
C. Train the Stage-4 detection LoRA adapter separately, on top of the base.
```

**Per in-scope stage (isolated):**
```
1. Construct the stage's input from the datasets (or a small hand-built set
   where no dataset provides it — see §7.1).
2. Run the local model (shared base + stage prompt; + detection adapter for
   Stage 4; PaddleOCR for OCR stages).
3. Score the output against ground truth for that stage.
4. Judge accuracy. Within tolerance → lock. Else → tune prompt / add adapter.
```

Stages are tested one at a time, in sequence. We do **not** feed one local stage's output into the next during benchmarking — each stage is judged on its own against ground truth, so a weak stage never contaminates another's score.

### 7.1 Ground-truth coverage (know this before starting each stage)

Not every stage has ready ground truth. This gates which stages can be benchmarked immediately vs. which need a label-build first.

| Stage | Ground truth | Status |
|---|---|---|
| **Stage 4 — Symbol Detection** | Gupta real boxes + Kaggle | ✅ Full — start here |
| **Stage 1.5 — OCR** | Tag transcriptions from Gupta crops | ⚠️ Build (readable without domain expertise) |
| **Stage 12 — Relation Validation** | PID2Graph edges (connectivity) | ⚠️ Partial |
| **Stage 10.5 — Skid Grouping** | PID2Graph grouping (sparse) | ⚠️ Partial / thin |
| **Stage 13 — Entity Validation** | None — "correct keep/correct/remove" is unlabeled | ❌ Build (needs P&ID judgment) or smoke-test only |
| **Stage 1 — Sheet Classification** | None — pages not labeled P&ID/legend/etc. | ❌ Build (visually obvious, easy) |
| **Stage 2 — Title Block** | None — metadata fields not labeled | ❌ Build (small, easy) |

For partial/absent stages: complete the ground truth by hand where feasible, adjust the testing parameter, or smoke-test only for the low-stakes ones. None of this blocks Stage 4, which has full ground truth — so **Stage 4 goes first** and the label-building for other stages happens in parallel or later.

**Definition of done (per stage):** the local model's accuracy against ground truth is within the target tolerance for that stage. **Project done:** all in-scope stages pass in isolation, on one shared task-neutral base VLM + a Stage-4 detection LoRA + PaddleOCR, with any needed stage adapters. Tying the stages together into a live local pipeline (and the transport/prompt-port work that requires) is a **separate later phase** — see §10.

---

## 8. Priority Order

1. **Stage 4 — Symbol Detection FIRST** (🟥 Full). It has full real ground truth (Gupta 72/20 + Kaggle) and it selects the shared base. Bake off Qwen3-VL vs InternVL3 vs Molmo against Gupta ground-truth boxes → domain fine-tune the winner (task-neutral) → train the Stage-4 detection LoRA on top. **This decision unlocks every other stage.** (Detailed execution plan: separate Stage 4 doc.)
2. **Stage 1.5 — OCR bake-off** (🟧): PaddleOCR vs docTR vs Qwen3-VL-OCR. Judge on tag-association accuracy (does each tag bind to the right symbol), not isolated CER. Build the tag-transcription ground truth from Gupta crops. Lock winner. (Independent track — not a VLM.)
3. **Reuse the base VLM at Stages 10.5, 12** (🟧): same checkpoint, stage-specific prompts, scored on PID2Graph grouping/edge ground truth. Add a LoRA adapter only if a stage underperforms.
4. **Stages 13, 1, 2** (🟩 / build GT): these have no ready ground truth — build a small labeled set (easy for 1 and 2; needs P&ID judgment for 13) or smoke-test only. Lower priority.
5. **Dynamic Cropping, Stage 5** (🟩, disabled by default): lowest priority — swap and smoke-test when reached.
6. **Confirm offline stages** (0, notes, 3, 6, 10, 11, finalize) unchanged.
7. **LAST (only if time + need):** small local LLM fallback for Notes Extraction on odd layouts.

**Deferred to a later phase (after isolated benchmarking is done — see §10):** tying the stages into a live local pipeline, per-stage prompt-porting to the chosen local model's format, and the Anthropic-format transport shim that makes the real agent call local models.

---

## 9. Notes on Disabled Stages

Stages 5, 12, and Dynamic Cropping are **disabled by default** in the current agent. They are still in scope (they *can* be enabled and *do* use cloud models), but they are **lower priority** — substitute and benchmark them after the always-on stages, since they don't affect the default pipeline output.

---

## 10. Deferred — Integration Phase (after isolated benchmarking)

This whole spec covers **isolated, per-stage benchmarking against ground truth**. The following are real work but are explicitly **deferred until benchmarking is done and models are selected** — they are not needed to benchmark a stage in isolation, and doing them early wastes effort on models that might not win.

| Deferred item | Why deferred | Trigger to start it |
|---|---|---|
| **Fine-tune the winning model for pipeline fit** | The downstream stages were tuned on Claude's output format; the local winner may format things slightly differently. Fix this on the *chosen* model, not all candidates. | After the base + per-stage winners are locked |
| **Per-stage prompt-porting** | The agent's prompts are calibrated for Claude (e.g. Stage 4's forced tool-call with an ontology enum). Local models have different tool-call dialects / JSON adherence — prompts need re-engineering per stage per model. Real work, but only for the selected model. | After base model is chosen |
| **Anthropic-format transport shim** | The agent speaks the Anthropic Messages API; vLLM doesn't natively. A LiteLLM Anthropic-passthrough shim is needed so the real agent can call the local model. Not needed while benchmarking (we call models directly). | When wiring the local pipeline together |
| **End-to-end chained run + interaction check** | Locally-optimal per stage ≠ globally-optimal pipeline. A stage that scores well in isolation can still degrade downstream when chained. | After all in-scope stages pass in isolation |

**Known deferred risk (named, not solved):** isolated ground-truth scores will not perfectly predict end-to-end performance. A local Stage 4 that scores well against Gupta boxes but formats detections differently than the downstream stages expect can still cause degradation when chained. This is expected and is what the integration-phase fine-tune (row 1 above) and the end-to-end check (row 4) exist to catch.
