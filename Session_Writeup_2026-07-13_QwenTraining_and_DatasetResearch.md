# Session Write-up — Qwen Domain-Base Training, Eval, v2, Benchmark Harness, and the Dataset World-Scan

**Date:** 2026-07-12 → 2026-07-13
**Scope:** everything from the launch of the first Qwen domain-adaptation training run through the
dataset deep-research and its conclusions. Written as the single narrative record of this phase;
individual artifacts (notebooks, eval numbers, checklist updates) live in the repo as usual.

---

## 0. Where we stood when training started

- **Stage 4 (detection) direction settled:** Molmo2-O-7B + 512px/2×-upscale/autocontrast tiling is
  the Stage 4 candidate (dense-sheet experiment showed ~2.3× F1 improvement over production-style
  1024px tiles; full-20 zero-shot F1 = 0.434 vs incumbent Claude 0.380, both below the 0.70 bar).
- **Nemotron ruled out** for symbol detection with a diagnosed root cause: given a symbol-grounding
  task it emits text-line/layout bands (its DocVQA training distribution), confirmed visually —
  ~90 wall-to-wall horizontal strips, zero boxes on actual symbols. Kept as a plausible candidate
  only for pure text-reading stages (1.5/2), tested elsewhere, not in this repo.
- **The two-base decision (recorded spec deviation):** the spec (§5) commits to ONE shared base
  chosen at Stage 4. We deviated deliberately: **Molmo2 owns Stage 4; Qwen3-VL-8B becomes the
  shared domain base for every other VLM stage** (1 sheet classification, 2 title block, 5 OCR
  tiebreaker, 10.5 skid grouping, 12 relation validation, 13 entity validation). Grounds: Qwen3-VL
  won the reasoning probe, and the spec itself anticipates reconsidering the base if the Stage 4
  winner underperforms on reasoning stages. Cost: two model lifecycles instead of one.

---

## 1. Infrastructure pivot: Google Drive → Hugging Face Hub

### The blocker
Colab's `drive.mount()` now uses an **ephemeral auth flow hard-bound to the account that owns the
runtime**. Since GPU quota (Colab Pro+) lives on the boss's account, every mount attempt sent a
2FA verification code to his phone — regardless of browser, incognito, or device. Confirmed
structural (same failure in Chrome and Safari): the account picker pre-fills the runtime owner's
email because the *runtime* initiates the auth, not the browser. There is no workaround from
inside the notebook.

### The fix — account-independent storage layer
- **Private HF dataset repo `timthy45/pnid-extraction-datasets`** (personal account): all three
  dataset zips uploaded once from the local Mac (Gupta 6.7GB, Kaggle 1.5GB, PID2Graph 9.3GB ≈
  16.3GB total) via `hf upload-large-folder` — resumable, hash-deduped, wrapped in a self-healing
  retry script. Verified on the Hub with correct sizes.
- **Private HF model repo `timthy45/qwen3vl-pnid-domain-base`**: training checkpoints, pushed
  every 200 steps.
- **Auth = one bearer token string.** No OAuth, no popup, no 2FA, no account coupling. Compute
  (boss's GPU) and storage (personal HF) are now fully independent systems.

### Why it's better than both alternatives
- vs **Drive:** no auth wall at all; data lands on the VM's local NVMe (fast reads — the Drive
  FUSE mount previously turned a 30k-file scan into 30+ minutes); checkpoints survive VM death
  because they're pushed off-machine every 200 steps.
- vs **local Mac:** no usable GPU for an 8B fine-tune; no persistence across lid-close. The Mac's
  correct role is one-time staging, which is exactly what it did.
- Honest caveat: the training *speed* is unrelated to this switch (GPU math is the same); the
  switch is about **being able to run at all** plus I/O and resilience.

A standalone fetch notebook (`Stage4_FetchDatasets_PersonalDrive.ipynb`) was also built for the
Drive path before the pivot; superseded in practice by the HF flow.

---

## 2. Training v1 — the overnight run and what it taught us

**Notebook:** `Stage4_QwenDomainBase_Overnight_Training.ipynb`
**Recipe:** Qwen3-VL-8B-Instruct + LoRA (r=16, all-linear), teacher-forced with prompt masking,
four task types mixed (~4,087 examples/epoch), 3 epochs, checkpoint+push every 200 steps,
exact-step auto-resume, per-step exception skip, 11h time budget. H100, ~0.68s/step → ~2.3h total.

**v1 task mix:**
1. **OCR — "list every tag on this tile"** (Gupta tiles, Tesseract pseudo-labels)
2. **Symbol counting** (Gupta real boxes, class-agnostic)
3. **Typed symbol summary** (Kaggle synthetic, 32 classes)
4. **Connectivity Q&A** (PID2Graph Patched OPEN100, 1500px patches, "are these two symbols
   directly connected?" — the only real GT matching stages 10.5/12's actual job; parser verified
   against the real zip before shipping)

**Mechanical issues hit and fixed:** `torchao` 0.10 vs peft incompatibility (uninstall, not
upgrade); edge-sliver tiles crashing Qwen's processor (aspect ratio > 200); a corrupted config
cell (token pasted into the assert); kernel expiry mid-diagnosis (recovered — checkpoints on HF).

### The suspicious signal, and the diagnosis
Training loss collapsed to ~0.02 within 200 steps and stayed there. The morning smoke test (n=1
per task) showed perfect **templates** with wrong **content**: count answered 0 vs true 24; OCR
emitted a degenerate repetition loop (`V002-11A-V002-11A-…`); typed summary got 3–4 of 7 classes;
relation was correct (coin-flip, n=1).

First hypothesis was a **masking bug** (near-zero loss + wrong generation on trained data usually
means the loss was computed over nothing). A model-free diagnostic reproduced the notebook's exact
masking logic on a real example: **masking was correct** — 13 unmasked positions decoding to
exactly the full target sentence, prefix match true. So the training was real; the model had
genuinely learned the templates and the dataset's statistical priors instead of visual grounding.

### The quantitative eval (the decisive evidence)
**Notebook:** `Stage4_QwenDomainBase_Eval.ipynb` — ~50 examples/task, raw base as control on
identical pools, slices chosen to minimize train overlap (Kaggle >1000, seed-999 relation pairs;
Gupta overlap unavoidable with 72 sheets and flagged as train-fit).

| Metric | Base | Adapter (v1) | Verdict |
|---|---|---|---|
| relation accuracy | 76.9% (37/50 undecided) | **90.0%** (0 undecided) | ✅ real win — most trustworthy number (seed-disjoint) |
| typed-summary class F1 | 0.00 | **0.36** | ✅ real learning (base's 0 partly a vocab artifact) |
| count: answered "0" | 6% | **100%** (true zero: 12%) | ❌ collapsed to the majority-class prior |
| OCR tag recall | **39.0%** | 1.95% | ❌ **net-destructive** — training on noisy list-all-tags targets overwrote Qwen's native reading |

Key lessons extracted:
- **Zero-skew teaches the prior.** Most Gupta tiles are empty margin; unchecked, "0" becomes the
  safe answer for every tile.
- **Templated targets are gameable.** One fixed phrasing per task lets the model drive loss to
  ~zero without reading the image; per-task loss curves (added in v2) expose this, a blended
  average hides it.
- **Noisy pseudo-labels can subtract capability.** The Tesseract list-all-tags task didn't just
  fail to teach reading — it *unlearned* it. Qwen's untuned OCR (39% agreement with Tesseract) is
  an asset to preserve, not retrain.
- Loss ≈ 0 early is a warning sign, not a success signal.

---

## 3. Training v2 — the corrected run (launched, in progress at time of writing)

**Notebook:** `Stage4_QwenDomainBase_Training_v2.ipynb` — same resilience machinery, fresh LoRA
(not resumed — v1's pathologies would fight the fixes), checkpoints to `v2/` so v1 stays
comparable.

**Changes, each tied to an eval finding:**
1. **Count rebalanced + de-templated** — zero-count tiles capped at ~15% of the task; six varied
   target phrasings (all digit-bearing so eval parsing still works).
2. **List-all-tags OCR dropped, replaced by single-tag reading** — tight crops around individual
   Tesseract-confident words (conf ≥ 60 only), "what does this label say?" This is the actual
   skill stages 2/5/13 use, on a learnable footprint, with the noisiest pseudo-labels excluded.
3. **Typed summary kept as-is** (it worked).
4. **Relation upweighted ~2×** (500 patches) — it serves the two stages with real ground truth.
5. **Per-task loss lines** in the training log; sliver-tile filter and torchao uninstall baked in.

**Early v2 signal (healthy, unlike v1):** pool = 6,850 steps/epoch, ~0.68s/step → ~3.9h for 3
epochs. Losses declining *gradually* — count 1.26→0.83, tag_read 1.6→~1.1, typed 2.2→~0.55,
relation ≈0.02 from the start (already-easy task). Gradual decline is what genuine learning looks
like here; v1's instant collapse was the tell that the task was gameable.

**Verdict pending:** run `Stage4_QwenDomainBase_Eval.ipynb` with
`ADAPTER_PATH_IN_REPO = "v2/latest"`. Watch `count.answered_zero_%` (should drop from 100 toward
~15) and whether tag-read beats the base instead of destroying it.

---

## 4. The all-stages benchmark harness

**`notebooks/all_vlm_stages_benchmarking/AllStages_Benchmark.ipynb`** — one parametrized notebook
(deliberately NOT one per model: copies drift and break comparability). `MODEL` config in cell 1
selects `gpt-5.5` (OpenAI API, low reasoning) / `qwen-base` / `qwen-domain-v1` / `qwen-domain-v2`
/ `molmo2` through an adapter layer (`generate(image, prompt) → text`); every model runs the
byte-identical harness. Results append to a shared CSV pushed to the HF dataset repo, so
cross-model comparison accumulates across runs; the final cell prints the matrix.

**The honesty table (what each score can actually claim):**

| Stage | Ground truth | Score meaning |
|---|---|---|
| 1 Sheet classification | none (all our pages are P&IDs) | recall-only; yes-to-everything scores 100% — smoke check |
| 2 Title block | none | raw extractions stored; only cross-model agreement computable |
| 5 OCR tiebreaker | synthesized | real accuracy on true-vs-corrupted-word picks; pseudo-GT from Tesseract conf ≥ 70 |
| 10.5 Skid grouping | sparse | not benchmarked; stage 12 is the nearest proxy |
| 12 Relation validation | **real (PID2Graph)** | accept/reject accuracy on injected true/false edges — most trustworthy number |
| 13 Entity validation | partial (Gupta) | keep/remove on real symbols vs empty-region decoys — existence only, typing untestable |

**On "GPT-5.5 is the incumbent":** a code-level investigation of the real agent found the Stage 4
path calls `claude-opus-4-8` via Anthropic's Messages API with forced tool-use — no GPT-5.5, no
`reasoning_effort`, no OpenAI SDK anywhere in that codebase. The user has separate information
that production actually runs GPT-5.5 at low reasoning; the harness therefore supports it as the
incumbent reference, explicitly labeled a **proxy** (our prompt + tiling, not a verified
replication of an unseen production config).

---

## 5. The dataset world-scan (deep research) — findings

**Method:** a fan-out/adversarial-verify research workflow (search agents per gap → fetch primary
source pages → 3-vote adversarial verification per claim). Run at max effort; **stopped by choice
at 56/103 agents** when live cost tracking (a per-minute proxy streamed from transcript volume)
showed it running heavier than expected (~1.1M transcript tokens; realistic cost estimate
$15–35). A cheap single-agent follow-up (~40K tokens) closed the remaining angles without
adversarial voting. Everything below survived verification or one confirming primary-source check.

### Genuinely new, usable, commercially-licensed
- **TextOCR** (Meta/FAIR, arXiv 2105.05486) — real images, ~900K–1M word annotations,
  rotated/curved polygon boxes + verified transcriptions. **CC BY 4.0 verified.** The Stage 1.5/5
  OCR transfer base we lacked.
- **DocLayNet** (IBM, HuggingFace) — 80,863 real human-annotated pages, 11 layout classes,
  sources include Manuals/Patents/Tenders. **CDLA-Permissive-1.0 verified** (a data-specific
  commercial-OK license). Strongest base for a Stage 1 sheet classifier.
- **CGHD** (DFKI, Zenodo 14042961) — **real** photographed hand-drawn electrical circuit
  diagrams: 3,173 images, **59 typed classes**, 245,962 boxes, rotation + text-string annotations,
  partial netlist connectivity. **CC-BY-4.0** per the authoritative Zenodo record. The only
  real+typed+boxed engineering-drawing set found — off-domain (circuits, not P&ID) but explicitly
  in the invited transfer tier.
- **Enginuity** (Predii/ORNL) — the Jan-2026 paper is a proposal (50K automotive diagrams,
  license unnamed = unverifiable), but the June-2026 release (`enginuity2025/enginuity-bench`) is
  real and downloadable: ~2.3K figures + parts tables from public-domain US military manuals,
  CC BY 4.0. Not P&ID; a transfer/pretraining asset.
- Form-extraction family for Stage 2: **SROIE** (permissive ✅), **CORD** (CC BY-SA — commercial
  OK but copyleft), **DocBank** (permissive but scientific-papers-only).

### License traps caught (the searches paid for themselves here)
- **Digitize-PID / Dataset-P&ID (Paliwal 2021): CC BY-NC-ND 4.0 — BLOCKED for commercial use.**
  The HuggingFace re-host (`digitize-pid-yolo`) shows *no license at all*; using it would have
  been exactly the NVIDIA-weights mistake repeated.
- **CGHD's GitHub badge says CC0-1.0 — the authoritative Zenodo data record says CC-BY-4.0**
  (attribution required). Same code-vs-data trap, caught; still commercially usable either way.
- **FUNSD and XFUND: confirmed non-commercial** — off-limits as training data.
- **SynthPID** (IIT Bombay, CVPRW 2026, arXiv 2604.16513) — newest find: 665 synthetic P&IDs
  topology-seeded from 12 real OPEN100 sheets, with per-node box+class AND GraphML connectivity;
  doubles as a generator. But the **data/code license is unverified** (repo 404s; the CC BY 4.0
  on arXiv is the *paper's* license). Flagged, not cleared.
- Roboflow P&ID sets ("P&ID Symbols" by PID Connect: 1,065 imgs / **181 typed classes**, CC BY
  4.0 self-declared; "P&ID Diagram": 43 imgs / 29 classes) — the most P&ID-specific typed boxes
  found anywhere, but **no Roboflow P&ID project discloses real-vs-synthetic provenance**.
  Confirmed industry-wide in the follow-up sweep: nobody states it. Images must be visually
  inspected before any of this counts as real ground truth.

### Confirmed dead ends (useful negatives — stop searching these)
- **ISA / ASME publish no open datasets** — paid standards documents only.
- **IEEE DataPort has zero P&ID datasets** (direct site search).
- **No university/lab-hosted P&ID corpora** outside the hosts already covered.
- **No synthetic generators beyond** Digitize-PID, pid_reader (GPL-3.0), SynthPID.
- **Stage 13 has no public dataset anywhere** with keep/correct/remove judgments on
  engineering-drawing entities. Closest structural proxies (all general-domain, licenses
  unverified): *Rechecked* (arXiv 2508.06556 — literal keep/correct/remove crowd loop),
  Cleanlab ObjectLab (method, not data), Jacquard V2, MJ-COCO-2025. Stage 13 data must be
  purpose-built; that's now a verified conclusion, not an assumption.

### The gap ledger after both passes
- **Fixed:** Stage 1.5/5 OCR base (TextOCR), Stage 1 classification base (DocLayNet).
- **Improved:** Stage 4 typing transfer (CGHD), Stage 10.5/12 (SynthPID as generator, pending
  license; CGHD partial netlists).
- **Still open:** Stage 4 real+typed P&ID data (the biggest gap — every candidate fails one axis),
  Stage 2 engineering-specific title blocks, Stage 13 (must be built).
- **Practical path for Gap 1:** inspect the PID Connect Roboflow set's pixels for provenance,
  and/or hand-type a subset of Gupta's real class-agnostic boxes in-house.

---

## 6. Why the flagship cloud model "succeeds" without any of this data — and what transfers to Qwen

The question came up: if no open P&ID data exists, how is the incumbent good at this? The honest
answer, grounded in what we measured:

1. **Diffuse web-scale exposure, not a curated dataset.** Frontier models absorb P&ID symbology
   incidentally (textbooks, forums, patents, catalogs). Teaches recognition, not precision.
2. **It isn't actually clearing the bar.** Measured this session: incumbent Claude F1 = **0.380**
   on real Gupta detection — below the project's 0.70 bar and below Molmo2's 0.434. The weakness
   is why this substitution project exists.
3. **Inference-time domain injection.** The production agent's real prompt is the domain
   knowledge: senior-engineer persona, symbol→entity cheat-sheet, five named
   hallucination-failure patterns, evidence rules, bbox-placement rules — written once by an
   expert, re-supplied every request.
4. **Pipeline resilience, not model brilliance.** Tiling + low-confidence recheck pass + failed-
   tile retry sweep + downstream validation let mediocre per-call accuracy survive.
5. **The "typed-correctly-on-real-sheets" claim is untested for everyone** — including the
   incumbent (CLAUDE.md's known permanent limitation).

**Transfer scorecard to Qwen — 4 of 5 mechanisms apply, 2 are unbuilt:**

| Mechanism | Status for Qwen |
|---|---|
| Broad pretraining exposure | Can't add post-hoc — the domain-adaptation LoRA (v2) is the deliberate substitute |
| Honest measurement discipline | Already in place (two-part metric, pass bar, no averaging) |
| **Engineered production prompt** | **NOT DONE** — all our zero-shot tests used simpler custom prompts. Highest-leverage, zero-GPU-cost item outstanding |
| **Recheck pass + retry sweep in the harness** | **NOT DONE** — pure harness engineering, no model change |
| Fine-tuning for domain knowledge | Underway (v2 running) |

---

## 7. The road to a definitive replacement (strategy recap)

Established earlier in the session and unchanged by anything since:
- Domain adaptation is the **foundation**, not the product. No 7K-example LoRA competes with a
  frontier model at general reasoning — and it doesn't have to. It has to win at ~8 narrow,
  repetitive jobs.
- **The bottleneck is labels, not compute.** Stages 1, 2, 13 have no ground truth; nothing trains
  or validates without it.
- **The highest-leverage move is distilling the incumbent:** run the production agent over
  hundreds of real drawings, capture every stage's input→output pairs, fine-tune one per-stage
  LoRA on the shared domain base against those teacher labels. "Agreement with the incumbent" is
  simultaneously the training objective and the spec's own acceptance criterion.
- Per-stage adapters only on measured failure; acceptance = per-stage benchmark vs incumbent
  output + one end-to-end pipeline comparison on held-out sheets.
- Stage 4 stays its own track (Molmo2 + tiling fix + detection LoRA on Gupta; documented Plan B
  detector if the fine-tuned VLM misses the bar).

---

## 8. Immediate next actions

1. **Eval v2** when training lands: `Stage4_QwenDomainBase_Eval.ipynb` with
   `ADAPTER_PATH_IN_REPO="v2/latest"`. Decision metrics: `count.answered_zero_%` ↓ from 100,
   tag-read ≥ base (39%), relation/typed hold or improve.
2. **Port the real production prompt** into the Molmo2/Qwen benchmark notebooks (zero GPU cost,
   plausibly the biggest single zero-shot mover left).
3. **Add recheck-pass + retry-sweep** to the scoring harness (mirrors the real pipeline's
   resilience; pure engineering).
4. **Run the all-stages benchmark** for `gpt-5.5` (incumbent proxy) and `qwen-domain-v2` —
   the CSV matrix gives the first honest head-to-head across stages 1/2/5/12/13.
5. **Gap-1 groundwork:** inspect the Roboflow PID Connect images for real-vs-synthetic
   provenance; scope the in-house typing of Gupta boxes.
6. **Strategic:** start capturing the production agent's per-stage outputs on real drawings —
   the distillation corpus everything else depends on.

---

## 9. Artifacts produced this session

| Artifact | What it is |
|---|---|
| `timthy45/pnid-extraction-datasets` (HF, private) | All 3 dataset zips + shared benchmark results CSV |
| `timthy45/qwen3vl-pnid-domain-base` (HF, private) | v1 checkpoints (`latest/`, `epoch_*`), v2 under `v2/` |
| `Stage4_FetchDatasets_PersonalDrive.ipynb` | Drive-based fetch (superseded by HF flow) |
| `Stage4_QwenDomainBase_Overnight_Training.ipynb` | v1 training (4-task, self-resuming) |
| `Stage4_QwenDomainBase_Eval.ipynb` | Quantitative base-vs-adapter eval, 50/task |
| `Stage4_QwenDomainBase_Training_v2.ipynb` | v2 training (rebalanced count, single-tag read, relation ×2) |
| `notebooks/all_vlm_stages_benchmarking/` | One-notebook multi-model stage benchmark + README honesty table |
| This document | Session narrative record |
