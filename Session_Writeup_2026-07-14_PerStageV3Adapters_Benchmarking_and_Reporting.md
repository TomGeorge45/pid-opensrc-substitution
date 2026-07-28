# Session Write-up — Per-Stage v3 Adapters, n=120 Benchmarking, HTML/PDF Reporting

**Date:** 2026-07-14
**Scope:** everything from finishing the v3 per-stage LoRA training run through building a
trustworthy (n≥100) benchmark against GPT-5.5 and the old general adapter, building an HTML
report + PDF, and the next agenda item (combined multi-adapter pipeline benchmark) that was
proposed but **not yet started**.

**How to use this document (read this first):** This is a handoff for continuing this exact
line of work in a fresh conversation with no memory of the long session that produced it.
Everything below is stated as fact because it was true at write-up time — but **treat every
claim about live state (checkpoint existence, training progress, file paths, HF repo contents)
as something to re-verify, not something to trust blindly.** State changes: the relation adapter
may have been resumed and finished since this was written, files may have moved, tokens may
need re-pasting. Before recommending or building on top of any specific fact here, check it
against the actual repo/HF/Colab state. **If you hit a genuine gap — something this document
doesn't cover, or something that contradicts live state — stop and ask Tom directly. Do not
guess, and never fabricate a number, a file's contents, or a "result" that wasn't actually
produced by running code.** That last rule was violated in spirit nowhere in this session, and
it needs to stay that way — several exchanges below exist specifically because a fabricated
example or invented number would have been indistinguishable from a real one to a casual reader.

---

## 0. Where this picks up from

Prior context (see `Session_Writeup_2026-07-13_QwenTraining_and_DatasetResearch.md` and
`Stage4_Checklist_Status.md` for the full backstory): Qwen3-VL-8B-Instruct was trained as a
**general, mixed-task** LoRA adapter (v1, then v2) covering four tasks at once — connectivity
Q&A, typed symbol summary, symbol counting, tag reading. v2 fixed v1's counting pathology but
**destroyed tag reading** (72%→0% on the exact crops it trained on) while connectivity/typing
held. Diagnosed leading suspect: `target_modules="all-linear"` LoRA rewrites the vision tower
with gradients from tasks that never reward glyph precision.

This session's starting conclusion, stated by Tom: **"general training is useless."** The
question became whether **dedicated, per-stage adapters** (one task each, vision tower excluded
from LoRA) would do better than both the general adapter and prompted zero-shot models. This
write-up covers building and testing that.

---

## 1. What got built

### 1.1 Training notebook — `notebooks/stage4/Stage4_PerStage_Stage13_and_Relation_v3.ipynb`

Trains two independent adapters back-to-back in one Run All, each auto-resuming from its own HF
checkpoint if interrupted, auto-skipping if already complete:

- **v3-stage13** (entity validation, keep/remove a proposed symbol location) — **NEW task design**
  with deliberate absence supervision: three decoy types (empty region, shifted near-miss,
  wrong-size box) mixed 50/50 with real boxes, specifically to prevent the "always say yes" bias
  that made the OLD v2 adapter score **35% (below chance)** on this exact task. Trained on Gupta's
  **72 TRAIN sheets only**. **Status: fully trained, all 3 epochs complete.**

- **v3-relation** (connectivity, stage 12/10.5) — fresh LoRA, bigger data than v2 (adds the
  previously-unused `Patched/Dataset PID` tree alongside `PID2Graph OPEN100`, schema-verified
  before use), 5 phrasing/answer-format variants instead of 1. **Status: INCOMPLETE — paused
  mid-epoch-0 at step ~44,853 of 64,911 when it hit its 8-hour time budget cutoff. Never
  resumed since.** (Verify current status before assuming this is still true.)

Both use **language-only LoRA**: `target_modules` discovered at runtime by inspecting
`model.named_modules()`, filtering `nn.Linear`, excluding anything with "visual"/"vision"/
"image_tower" in the name — the direct fix for the all-linear hypothesis above. Confirmed
included: `['down_proj','gate_proj','k_proj','lm_head','o_proj','q_proj','up_proj','v_proj']`.

Checkpoints live at `timthy45/qwen3vl-pnid-domain-base` on HF: `v3-stage13/latest`,
`v3-relation/latest` (plus the old `v2/latest`, 210MB, kept for comparison).

### 1.2 Benchmark notebook — `notebooks/all_vlm_stages_benchmarking/PerStageV3_Stage13_Relation_vs_GPT55.ipynb`

Grew over the session to 25 cells / 11 sections. Loads Qwen3-VL-8B once, attaches
`v3-stage13`, `v3-relation`, and (Section 11) `v2` as three separate named PEFT adapters on the
same base model instance — switching between them (`model.set_adapter(...)`, or
`model.disable_adapter()` context manager for the untouched base) rather than reloading.

Sections, in order: (1) Config, (2) Install, (3) Data fetch, (4) Build eval pools (stage13 +
relation, n=120 each), (5) Model callers (base load + adapter attach + `gpt_generate`/
`qwen_generate`), (6) Scoring functions, (7) Run all six scores, (8) Verdict table + push to HF,
(9) Text extraction / OCR tie-break eval (n=65, reusing yesterday's exact task definition),
(10) Qualitative sample gallery (real crops + every model's raw answer, saved as standalone
HTML + auto-download), (11) V2 general adapter re-scored on the same n=120/65 pools.

**Section 11's actual results were never confirmed run in this session** — it was added late;
check whether it's been executed before assuming the "V2 (today)" numbers exist anywhere.

### 1.3 Results (the numbers that ARE confirmed, all from real runs pasted by Tom)

| Task | GPT-5.5-low | Qwen base | Dedicated v3 adapter | V2 adapter (yesterday, n=25) |
|---|---|---|---|---|
| Entity validation (stage 13) | 66.7% | 45.0% (below chance) | **89.2%** | 35.0% |
| Relation validation (stage 12/10.5) | 72.5% | 80.0% | **89.2%** (partial, 1/3 epochs) | 84.0% |
| Text extraction (stage 5, OCR tie-break) | 98.5% | **100.0%** | n/a (no adapter built) | 66.7% (u16/25 — mostly unparseable) |

n=120 for entity/relation, n=65 for text extraction (OCR tie-break pool naturally capped by how
many qualifying text regions exist across the 20 Gupta TEST sheets — not a bug, that's just how
many were found). "u16/25" etc. = undecided count out of total; high undecided = low-confidence
result, not a real capability signal (this is why yesterday's GPT-5.5 numbers in the historical
table are mostly meaningless — token-starvation bug, `max_tokens=64` before a later fix).

**Qualitative confirmation (real, not fabricated — see Section 1.10):** the actual crops show
`qwen-base` inventing reasons to reject real symbols on stage13 (e.g. claiming a box "contains
only the red text 'Fail Last/D'" when ground truth is `keep`) — a concrete illustration of the
below-chance bias that absence-supervision training fixed.

### 1.4 Datasets & train/test split (the exact framing used in the report — reuse this, don't re-derive)

- **Gupta P&ID** (real, confirmed genuine CAD-drawn engineering drawings — see §1.9): 92 sheets,
  **72 train / 20 frozen TEST** (`test_ids.json`, never touched by training). Source for entity
  validation (stage 13) and text extraction (stage 5) — both training and every eval draw
  exclusively from the correct split.
- **PID2Graph** (real): two trees, `PID2Graph OPEN100` (≤1,600 patches) and `Dataset PID`
  (≤4,000 patches, schema-verified before use). Source for relation validation. **Important
  caveat: eval is seed-disjoint (seed 8181 vs training's 777), NOT file-disjoint like Gupta.**
  Same source trees for train and eval — a real methodological gap, already flagged in the
  report, worth closing if this becomes load-bearing for a bigger claim later.
- **Kaggle P&ID symbols** (synthetic, 32 classes): only used by the OLD general v1/v2
  typed-summary task — irrelevant to anything in the v3/summary table above.
- **V2 (general) adapter**: trained on all four general tasks mixed together, evaluated on
  ~50-example held-out-leaning pools — looser methodology than v3's frozen split.

### 1.5 HTML report (published Artifacts — private, URLs only work for Tom's account)

Two versions exist, both still live:
- Plain version (no diagram): `https://claude.ai/code/artifact/122f42bd-6515-4f59-8659-bef99db73f76`
- **Preferred/final version** (centered title + pictorial diagram, chosen by Tom as "perfect"):
  `https://claude.ai/code/artifact/2230e2da-70cc-48dd-9dcf-4156b85f8fe3`

The preferred version's structure, top to bottom: centered `<h1>` title "Task-based LoRA
adaptors benchmarking" → **"What changed" diagram** (Yesterday panel: one always-attached v2
adapter serving 4 tasks at once, shown with the actual mixed result; Today panel: **three
self-contained boxes, each with its OWN embedded mini "Qwen3-VL-8B base" box** — not one shared
base box outside — showing v3-stage13, v3-relation, and the no-adapter/base-wins path for
reading, each with real input→output shape and score) → Summary table → Chart (grouped bars,
validated categorical palette, hover tooltips) → Datasets & train/test split (trimmed to only
the 3 tasks actually in the summary table — Kaggle bullet deliberately removed since it's not
one of them) → **"Next" section** (its own card, dashed border, "not built yet" badge — one
input flowing through one VLM instance with adapters attaching/detaching in sequence, merging
into one combined output, then benchmarked — explicitly framed as future work, this is what
Section 2 below is about) → Examples (12 real crops with real model outputs, see §1.10) →
two historical progress tables pulled from the earlier `stage4-status.html` scratchpad file
(base→v1→v2 metrics, and the four-trained-tasks before/after breakdown).

Source file: `/private/tmp/claude-501/-Users-tomgeorge-pid-ml/ab9768a5-fde5-408a-ac4f-58f1394f263c/scratchpad/per_stage_v3_report_pictorial.html`
— **note this path is inside a session-specific scratchpad; it will NOT exist in a fresh
session/sandbox.** If continuing report edits, either ask Tom to re-share the file or rebuild
from the artifact URL content (fetch it, don't recreate from memory).

### 1.6 PDF export

Generated via headless Chromium (Puppeteer, installed fresh via npm since no browser was
available locally) — necessary because the chart is drawn by inline JavaScript at page-load
time, so a non-JS HTML→PDF converter would render a blank chart. Rendered A4 landscape,
1100×900 viewport. Final file copied to `/Users/tomgeorge/Downloads/per_stage_v3_report.pdf`
(7 pages, ~1.3MB). All 7 pages were visually verified before handing off.

### 1.7 The Xet CDN saga (long, but the fix matters for every future Colab session)

Recurring error all session: `HTTPStatusError: 403 Forbidden ... SignatureError: invalid key
pair id` on Hugging Face's Xet-backed CDN (`xet-bridge` URLs), hitting the base Qwen model
download, adapter checkpoint downloads, and dataset zip downloads — at different times, in
different notebooks, on different files (both private repos AND the public
`Qwen/Qwen3-VL-8B-Instruct` repo). **Confirmed server-side, not client-side**: fully uninstalling
`hf_xet` (verified via `pip show hf_xet` → not found) did NOT stop the Hub from routing through
the same Xet-bridge CDN path and hitting the same error.

**What was tried and rejected:**
- `HF_HUB_DISABLE_XET=1` env var — no effect (likely set after `huggingface_hub` was already
  imported, or simply doesn't override server-side routing for Xet-only-stored content).
- **Google Drive as a cache/bypass** — flatly ruled out by Tom: `drive.mount()` needs interactive
  browser OAuth every fresh runtime, which breaks unattended/overnight runs. Do not suggest Drive
  again unless Tom brings it up.
- **ModelScope as an alternate mirror for the public Qwen weights** — tried, confirmed to exist
  (`Qwen/Qwen3-VL-8B-Instruct` is dual-published there), but **measured ~1Mbps from this specific
  network vs ~200Mbps on HF** — a 200x penalty that makes it strictly worse here despite dodging
  the Xet bug (17GB at 1Mbps ≈ 37 hours). Also: a `ThreadPoolExecutor`-based timeout used to
  "abandon" a slow ModelScope download couldn't actually kill the background thread (Python can't
  force-kill threads), so the abandoned download kept running and its log lines bled into
  whatever cell ran next, causing real confusion (looked like unrelated cells were mysteriously
  slow). **ModelScope was fully removed from both notebooks. Do not re-add it as a fix for this
  network's users — it's a proven net negative here.**

**What actually works and is now baked into every download path in both notebooks:** a hardened
retry loop — up to 20 attempts, exponential backoff (10s → 90s cap) with jitter, and critically
**no `force_download=True`** (that wastes bandwidth restarting already-good partial downloads;
HF re-resolves a fresh signed URL on every call regardless, so plain retry gets the same benefit
for free). Every observed real failure this session cleared within ≤7 attempts. If this bug
recurs and the existing retry budget isn't enough, raise `max_attempts`, don't reach for
ModelScope or Drive again.

### 1.8 Other recurring gotchas worth knowing before touching these notebooks again

- **VS Code buffer overwrite**: when a notebook is open in VS Code while Claude edits it via
  `NotebookEdit` (or raw JSON), VS Code's in-memory buffer can re-save over the edit —
  specifically, a scrubbed HF token in the Config cell reappeared as the real token **twice**
  after being cleaned. Always re-`grep` for `hf_[A-Za-z0-9]{20,}` / `sk-[A-Za-z0-9_-]{20,}`
  after any notebook edit, right before publishing/committing, even if you scrubbed it minutes
  ago in the same session. None of these leaks were ever pushed to GitHub (verified via
  `git log -p` + `git show :<path>` against the staged index every time) — but don't assume
  that'll stay true; keep checking.
- **Colab vs VS Code confusion**: Tom edits/views the `.ipynb` files locally in VS Code, but
  actually *runs* them in a separate browser tab at colab.research.google.com (GPU there, not
  locally). `/content/...` paths only exist in that live Colab session, never on the Mac or in
  VS Code. Files downloaded via `google.colab.files.download(...)` land in the Mac's normal
  Downloads folder via the browser — but can silently fail if a popup/download blocker catches
  it; check `/Users/tomgeorge/Downloads/` and tell Tom to watch for a blocked-download icon in
  the browser address bar if a file doesn't show up.
- **Colab runtime resets happen often and silently**: several times this session, cells failed
  with `NameError` for things that were definitely defined earlier (`random`, `csv`, `N_PER_TASK`,
  `GUPTA_RAW`) — each time, root cause was the Colab runtime having restarted (or Tom running a
  cell in isolation without the earlier setup cells in that session). When this happens, nothing
  from Sections 1 onward is in memory; the fix is always "rerun Section 1 → whatever section you
  actually need," not a code fix. Sections were written to be resumable/skippable but **cells
  still assume everything above them in the SAME notebook has run in the CURRENT kernel** — they
  are not fully standalone even when logically independent (e.g., Section 9's OCR eval doesn't
  need Section 4-8's stage13/relation pools, but does need Section 5's `model`/`gpt_generate`).
- **n=25 is proven unreliable on this project.** The exact concrete example: relation validation
  showed v2 adapter "beating" prompted base 84% vs 80% at n=25 — a single flipped answer would
  erase that entire margin. All serious claims from here on use n≥100 (practically n=120, or
  n=65 when a task's eligible pool is naturally smaller, like OCR tie-break).
- **GPT-5.5 needs `max_completion_tokens=2000`, not small values.** A prior run at `max_tokens=64`
  caused GPT-5.5 to exhaust its token budget on internal reasoning before writing an answer,
  returning empty strings that scored as "undecided" — not a real capability signal. Yesterday's
  historical n=25 table (still shown in the report) has this exact artifact baked into it
  (`u20/25`, `u25/25`, `u13/25` on GPT-5.5's rows) — don't mistake those numbers for genuine
  low performance; today's fixed-harness GPT-5.5 numbers in the main summary table are the
  trustworthy ones.

### 1.9 A tangent worth remembering: dataset provenance sanity-check

Tom asked whether the entity-validation example crops looked "too perfect" to be real P&ID
sheets. Checked by opening an actual full test sheet locally
(`notebooks/stage4/dense_sheets_sample/151.jpg`) — confirmed it's a genuine, correctly-drawn
P&ID (color-coded pipe classes, real instrument tag bubbles, proper title block reading
"PIPSampleProject / Sample PIP Project, San Rafael CA", drawing number `PIP-01-101`). Real
engineering drawing conventions, not synthetic/stock clipart — but explicitly a publicly-
circulated **sample/starter** CAD file, not a live proprietary industrial P&ID, which is normal
and expected for an open academic dataset (real confidential P&IDs are essentially never
released publicly). One example crop had a partial watermark suggesting it came from a
different, watermarked sample sheet among the 20 — not evidence of fabricated data, just
consistent with these being freely-circulated demo files.

### 1.10 Integrity note on the qualitative examples

The 12 example cards in the report (real crops + real model answers) came from code Tom
actually ran in his live Colab kernel (`Section 10` of the benchmark notebook), which he
downloaded and shared back. **A card-extraction bug on the first attempt at merging that file
into the report corrupted the HTML** (non-greedy regex got confused by nested closing `</div>`
tags, causing card 2's content to render squeezed/garbled with character-by-character text
wrapping) — this was a rendering/extraction bug, not fabricated data; it was diagnosed (proper
tag-depth-tracking extraction instead of regex-guessing) and fixed, verified by checking
div-open/close balance before republishing. If the Examples section ever looks visually broken
again, suspect the same class of bug first (malformed nested-tag extraction), not the
underlying data.

---

## 2. What's proposed next (agreed direction, but genuinely NOT STARTED — no code exists yet)

Tom's ask, verbatim in spirit: build a benchmark that combines multiple pipeline steps
together — ideally run the real production agent as-is with GPT-5.5, and separately run the
adapter-switching method (Qwen + all local adapters + Molmo2) end to end, compare which does
better, and from that decide what (if anything) still needs more fine-tuning.

**Three real gaps identified, laid out to Tom, awaiting his input on the first:**

1. **Prod-agent access is unconfirmed.** The real `pnid-intelligence-agent` is a separate
   codebase; this repo only has code-verified *facts* about it (`Agent_Pipeline_Facts.md` —
   tiling params, output schema, entity ontology is runtime/per-tenant not fixed). Whether Tom
   can actually run that agent end-to-end, and whether it can be pointed at GPT-5.5 instead of
   its real Claude/Google-Vision calls, is unknown. **Ask Tom this before building anything that
   assumes prod-agent access.** Fallback if the answer is no: build the same pipeline *shape*
   (detect → validate entities → validate relations) with GPT-5.5 doing every step, honestly
   labeled as "GPT-5.5 pipeline," not "prod."

2. **Combined ground truth only exists on PID2Graph** (symbols + edges together); Gupta has
   boxes but no relation GT. An end-to-end pipeline eval (detect → validate → relate → score
   against a real graph) should run on PID2Graph sheets. Caveat to build in from the start: the
   relation adapter *trained* on PID2Graph's OPEN100+DatasetPID trees, so this eval needs a
   genuinely **file-disjoint** holdout (not just the seed-disjoint split used for the isolated
   relation-task eval above) — a real methodological upgrade needed here, not a rehash.

3. **v3-relation is still only 1/3 epochs trained** (see §1.1 — verify current status). If it's
   the weak link in a combined-pipeline score, finishing its remaining ~2 epochs (resume via the
   training notebook's existing auto-resume) is the obvious lever — worth doing regardless of
   the combined-pipeline plan, independently.

**Separately, Tom's boss sent a compute-cost directive** (paraphrased): do CPU-only
preprocessing (dataset download/extraction, OCR, pool-building) **locally on the Mac**, zip the
prepared result, upload it (HF, not Drive — Drive's interactive auth already ruled out per
§1.7), and have the paid GPU Colab session do *only* the GPU-dependent steps (model load,
train/infer, push results) before immediately auto-disconnecting
(`from google.colab import runtime; runtime.unassign()`). **This was discussed and agreed as
the right direction but zero code has been written for it yet.** It's a real architecture
change across all three notebooks (training + 2 benchmark notebooks), not a one-line fix —
scope it explicitly with Tom before starting (which notebook first, etc.) rather than assuming
it applies everywhere silently.

**Proposed agenda order (as last given to Tom, not yet started):**
1. Get Tom's answer on prod-agent access (gap 1 above).
2. Build the local (Mac-side) prep script per the boss's directive — bundles in the
   file-disjoint PID2Graph holdout selection needed for gap 2.
3. Build the combined-pipeline notebook: one input → Molmo2 detects → Qwen+stage13 validates
   each detection → Qwen+relation judges pairs → assembled graph output, with adapters
   switching in-place on one Qwen instance, ending in `runtime.unassign()`.
4. Same pipeline with GPT-5.5 at every step (or the real prod agent, depending on Tom's answer
   to gap 1).
5. Score both arms on the same file-disjoint holdout, produce the comparison + report, and use
   the per-stage error breakdown to decide where further fine-tuning is actually warranted.

---

## 3. Working-style notes (so the next session doesn't have to re-learn these)

- Tom wants **terse, direct** answers and dislikes unnecessary padding or hedging once a
  direction is clear — but values honest caveats (undecided counts, seed- vs file-disjoint,
  "this wasn't actually run yet") over confident-sounding but unverified claims.
- Strong preference for **acting over lengthy back-and-forth planning** once the shape of a task
  is clear — but genuine architecture decisions (which notebook to touch first, whether an
  ambiguous instruction means X or Y) should be asked, not assumed.
- **Never fabricate a number, example, or file's contents.** This came up explicitly more than
  once (the qualitative gallery had to wait for a real file rather than a plausible-looking
  invented one; a "yesterday" OCR result had to be tracked down to its real n=25/artifact-count
  rather than trusted from memory). If real data isn't available, say so and ask for it — don't
  approximate it into something that reads as real.
- Wants **n≥100 (practically 120)** before treating any accuracy number as decision-grade.
- Explicit standing rule from Tom: **no Google Drive**, ever, for this project's Colab workflow
  (interactive auth breaks unattended runs) — HF is the shared storage layer instead.
- Notebooks should be written so a single **Run All** auto-resumes/auto-skips already-completed
  phases without the user needing to track manually which cell died.
- Visual reporting (HTML artifacts with real charts, PDFs) is valued and used to present
  findings, not just raw printed tables — but every chart must be built from real, already-
  computed numbers, following the repo's `dataviz` skill conventions (validated categorical
  palette, direct value labels, hover tooltips, table-view fallback).

---

## 4. Quick-reference: file/URL index

- Training notebook: `notebooks/stage4/Stage4_PerStage_Stage13_and_Relation_v3.ipynb`
- Benchmark notebook: `notebooks/all_vlm_stages_benchmarking/PerStageV3_Stage13_Relation_vs_GPT55.ipynb`
- Yesterday's 3-stage notebook (source of OCR tie-break logic + v2/base/GPT-5.5 n=25 numbers):
  `notebooks/all_vlm_stages_benchmarking/ThreeStages_GPT55_vs_QwenV2_vs_QwenBase.ipynb`
- HF dataset repo: `timthy45/pnid-extraction-datasets` (Gupta/Kaggle/PID2Graph zips, results CSVs)
- HF model repo: `timthy45/qwen3vl-pnid-domain-base` (`v2/latest`, `v3-stage13/latest`,
  `v3-relation/latest`)
- Report artifact (final/preferred): `https://claude.ai/code/artifact/2230e2da-70cc-48dd-9dcf-4156b85f8fe3`
- PDF: `/Users/tomgeorge/Downloads/per_stage_v3_report.pdf`
- Facts about the real production agent (tiling, schema, ontology caveats):
  `Agent_Pipeline_Facts.md`
- Overall project rules/scope: `CLAUDE.md` (note: strictly scoped to "Stage 4 only" as written —
  this entire session's per-stage work on stages 5/12/13 is a deliberate, Tom-directed deviation
  from that stated scope, not an oversight; keep doing it, but don't be surprised the top-level
  doc doesn't mention it yet).
