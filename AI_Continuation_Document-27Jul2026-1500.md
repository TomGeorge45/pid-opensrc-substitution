# PROJECT CONTINUATION DOCUMENT
## 2026-07-27, 15:00

---

### 1. PROJECT IDENTITY

- **Project Name:** PID-ML — local-substitution project for `pnid-intelligence-agent` / `pnid-extraction-agent` (the real, deployed P&ID document-intelligence pipeline at Rive).
- **What This Project Is:** Replaces each cloud-model call (Claude Sonnet/Opus, GPT-5.5, Google Vision) in the real agent with the best **local** VLM/LLM/OCR substitute, proven equivalent via benchmarking, one pipeline stage at a time.
- **Primary Objective:** Prove a local model stack matches cloud-model quality on Stage 4 (Symbol Detection) first, then the relationship/connectivity stage — the two stages this session's work covers.
- **Strategic Intent:** Cut cloud-model dependency/cost in production without a quality regression. Stage 4's winner becomes the **shared base VLM** every later stage reuses via prompting alone.
- **Hard Constraints (from `CLAUDE.md`, do not re-litigate):**
  - No trained-from-scratch detector architectures (YOLO/RT-DETR/etc.) — local VLMs/LLMs/OCR only, unless the documented Plan B is formally triggered.
  - Never average detection and typing scores into one number; never use recall alone as a pass bar.
  - Test-set discipline: frozen `test_ids.json`, zero train/test leakage, checked before every run.
  - No MLflow — `results.csv` + `experiments/stage4/v*.md` instead.
  - GPU/CPU split convention: CPU does prep/local/HF work; GPU sessions do only train/infer, then `runtime.unassign()`.
  - No Google Drive, ever — HF is the only shared storage for Colab workflows.
- **Project Memory Files (read these too):**
  - `/Users/tomgeorge/pid-ml/CLAUDE.md` — the authoritative Stage 4 project rules (above).
  - `/Users/tomgeorge/.claude/projects/-Users-tomgeorge-pid-ml/memory/MEMORY.md` — auto-memory index; notable entries: `user_is_tom.md` (the user IS Tom, not a third party to "report to"), `project_no_google_drive.md`, `feedback_gpu_cpu_split.md`, `project_prod_model_gpt55low.md` (**prod's real model is GPT-5.5-low, not the repo's Claude Sonnet-4-6 default**), `project_agent_history.md`.
  - **Important scope note:** `CLAUDE.md` states the repo is "currently scoped to Stage 4 only." In practice, this session's entire body of work — the entity-extraction multi-arm architecture and the full 3-pipeline relationship-stage benchmark — goes **beyond that stated scope**. This is real, deliberate, Tom-directed work, not a mistake, but the next AI should know `CLAUDE.md` has not been updated to reflect it (see Section 4, "Promote to project memory").

---

### 2. WHAT EXISTS RIGHT NOW

- **Repo state:** git branch `main`. Latest commit `5311211` "Add automatic retry-with-force_download to the data-fetch cell" — **this and the prior 4 commits are all Stage-4/adapter-training work from 2026-07-14, unrelated to this session.** Everything this session produced is **uncommitted and untracked** (`git status` confirmed 2026-07-27): `src/relation_bench/` (the entire relationship-benchmark codebase), `notebooks/e2e_harness/` (Colab notebooks), `Benchmark_Gaps_Register.md`, plus several other untracked planning docs (`Conversion_Layer_Plan.md`, `E2E_Harness_Plan.md`, `Extraction_Agent_Local_Plan.md`, `personal.md`, `personal2.md`, `scripts/`, `src/e2e_bench/`, `src/extraction_local/`). **Nothing from this session has been committed** — the user's standing rule is "only commit when explicitly asked."
- **Does it run right now?** Not verified as a single pipeline. Individual pieces were run and produced real output this session (see Section 4) — `src/relation_bench/*.py` modules import and execute correctly (confirmed live, multiple times). No test suite exists for this code; validation was empirical (self-tests + real-data runs), not `pytest`.
- **What is built and working:**
  - Full relationship-stage benchmark harness (`src/relation_bench/`): PID2Graph GT loader/contraction, R1 (raster line tracer), R2a (deterministic connectivity builder, now with the backbone-pass upgrade), R2b (hierarchy), `agreement_diff.py` (LLM-claims-vs-geometry comparison), `score.py` (P/R/F1 scoring harness), `relationship_pipeline.py` (new this session — full pipeline over given entities, both configs).
  - Real GPT-5.5-low extraction results for the 3 dev sheets (2026-07-24 rerun; **just pushed to HF this turn**, see below — the only copy previously existed only in this session's ephemeral scratchpad and would have been lost).
  - The `notebooks/e2e_harness/Pipeline3_R4_Validation_Benchmark_GPUOnly.ipynb` GPU notebook — ran successfully as a smoke test (8 candidates/sheet).
  - A published artifact with the full findings: **https://claude.ai/code/artifact/3b4ea58f-f77f-4a6c-9188-23cd04ed8aa2** ("Relationship Pipeline — Three Architectures Compared") — this is the single best summary of everything below; read it early.
- **What is partially built:**
  - Tier 1 upgrade #1 (off-page claim partitioning) and #4 (line-label filtering) — built, self-tested, run once on real data, but only manually adjudicated on 1 of 3 sheets (PX-2368).
  - The 12-sheet PID2Graph OPEN100 backbone-pass benchmark is complete; the 500-sheet Dataset PID (synthetic, dense) tree was **attempted and abandoned** — R1's raster pipeline could not finish a single Dataset PID sheet in 5+ minutes (images are ~7168×4561px, 7x OPEN100's pixel count; R1 was ported without the downscale step production has). Dense-sheet stratification remains untested for the backbone pass.
- **What is broken or blocked:**
  - Tier 1 upgrade #3 (symbol-extent resolution — box the real drawn shape, not the tag-text label) is **blocked**, not just unstarted. Root cause: PDF's `closePath` flag is `False` on every one of 5,608 tested vector paths (even genuinely closed shapes), and path bounding-box sizes span sub-pixel text glyphs to full-page rects — no reliable geometric signal separates "equipment outline" from "pipe segment" from "text stroke." Needs a real design decision (connected-component clustering? raster flood-fill? accept the crude bbox-inflation workaround permanently?), not a quick fix.
  - The local relation-validator (Qwen3-VL-8B + v3-relation adapter) is **confirmed degenerate as R4** — kept 0/8 candidates on all 3 real sheets (rejected everything, "No." repeated). This mirrors Probe 2's finding exactly, now reproduced inside the actual pipeline stage.
- **What has NOT been started yet:**
  - The full (492-candidate) R4 Qwen validation run — deliberately **not** run past the smoke test; see Section 4 for why.
  - A GPT-5.5-low R4 arm (the "does a competent validator help" question) — proposed, not built.
  - Manual recall ground-truth tracing for the other 2 of 3 real sheets (only PX-2368 was hand-traced).
  - Writing the full 2026-07-25 verification (R4-degenerate + recall trace + delta adjudication) into `Benchmark_Gaps_Register.md` — **this was offered to the user and never confirmed/done.** It currently exists ONLY in the published HTML artifact and this conversation, not in the repo's own markdown docs. This is a real gap — flagged explicitly below.
  - The 2-3 hour human annotation task (real edge/hierarchy/kind ground truth on AG/RIVE sheets) that `Benchmark_Gaps_Register.md` Group 2 has been asking for since 2026-07-23 — still not done. Without it, no real F1 can ever be computed on production-representative sheets; the "recall %" number in this session is an Opus-traced substitute, not certified GT.

---

### 3. ARCHITECTURE & TECHNICAL MAP

**Tech stack:** Python 3.12, `.venv-e2e` (local venv at `/Users/tomgeorge/pid-ml/.venv-e2e`, has `openai`, `huggingface_hub`, `fitz`/PyMuPDF, `PIL`, `numpy` — used for all local/CPU work this session). GPU work happens in Colab notebooks. HF (`timthy45/pnid-extraction-datasets` dataset repo, `timthy45/qwen3vl-pnid-domain-base` model repo) is the only shared storage. Real API keys live in `/Users/tomgeorge/pid-ml/.env` (`OPENAI_API_KEY`, `HF_TOKEN`, `GH_TOKEN`) — never paste these into docs; the notebooks currently have `HF_TOKEN` hardcoded in plaintext (pre-existing pattern in this repo, not introduced this session).

**Two parallel architectures exist in this project — do not conflate them:**

1. **Entity-extraction stage (multi-arm architecture).** Documented in `E2E_Harness_Plan.md` and the artifact's Column 1/2/3 comparison. Key idea: Arm 0 (regex), Arm 1 (whole-page VLM read), Arm 2 (Molmo2 points + Qwen reads the crop), Arm 3 (CV-hybrid) — **unioned** together, which measurably beat any single arm (e.g. one sheet went 0.140→0.780 revR). Decision D-M1 (in `E2E_Harness_Plan.md`): Molmo2 emits points with no text, so its detections are paired with the nearest OCR word within a radius (`tag_matching.py`) to get both a real shape location AND a real name — this is the answer to "why not just use Molmo2's coordinates for symbol-extent," a question the user asked and I initially answered wrong before correcting myself (see Section 4).

2. **Relationship-extraction stage (3-pipeline architecture) — this session's main focus.** Documented in the published artifact (URL above) and `Benchmark_Gaps_Register.md`.
   - **Pipeline 1** (`pnid-extraction-agent`, in production): OCR reasoning → ISA loop grouping (regex, free) → hierarchy pass (LLM) → cycle-break (deterministic) → connectivity pass (LLM, no geometry). No line-tracing exists in prod at all.
   - **Pipeline 2** (`pnid-intelligence-agent`, shelved by leadership, real code in PR #711): Stage 4 detection → Stage 6 line-tracing (vector-first, raster fallback) → Stage 10.5 skid grouping (LLM) → Stage 11 graph construction + ISA rules (deterministic) → Stage 13 entity validation (local adapter) → Stage 12 relation validation (local adapter, confidence-gated).
   - **Pipeline 3** (proposed, this project's own port, THE ONE THIS SESSION UPGRADED): given entities → R1 (line-graph, ported from Stage 6) → R2a (deterministic connectivity, ported from Stage 11) → R2b (hierarchy) → R3 (entity validation, Qwen+v3-stage13) → R4 (relation validation, Qwen+v3-relation) → R5 (semantic enrichment).

**Key files this session built/modified, and what each does:**
- `src/relation_bench/pid2graph_gt.py` — loads PID2Graph graphml GT, contracts connector/crossing chains into direct symbol↔symbol edges (SYMBOL_CLASSES = `{valve, instrumentation, tank, pump, general}`; PASSTHROUGH_CLASSES = `{connector, crossing, arrow}`).
- `src/relation_bench/graph_construction/path_traversal.py` + `build_relations.py` — **modified this session** to add the backbone pass (Tier-1 #2): BFS now optionally continues walking through symbols whose class is in `passthrough_symbol_classes` (e.g. `{valve, instrumentation, pump}`) instead of always stopping at the first symbol reached. Purely additive — old behavior unchanged when the new params are omitted.
- `src/relation_bench/agreement_diff.py` — **modified this session** to add claim partitioning (Tier-1 #1: off-page LLM claims are now counted explicitly via `OffPageClaim`, not silently dropped) and line-label filtering (Tier-1 #4: tags typed `"line"` are excluded from `_inflate_equipment_bboxes`'s output, so a pipe-spec label like `6"(300#)` can never be a tracer endpoint).
- `src/relation_bench/relationship_pipeline.py` — **new this session.** The full relationship pipeline over GIVEN entities (isolated mode), toggling `upgraded=True/False` to run original vs upgraded Pipeline 3. `PASSTHROUGH_TAG_TYPES = {valve, instrument, fitting, safety_device}` for real-drawing Tag.type values (this is a FROZEN mapping decision, flagged for Tom's review, mirroring `type_vocab.py`'s existing pattern — not yet reviewed).
- `src/relation_bench/score.py` — P/R/F1 scoring harness, stratified by line_type, endpoint-class pair, and sheet density group. Unmodified this session, used heavily.
- `notebooks/e2e_harness/PartB_Qwen_RelationRun_GPUOnly.ipynb` — the notebook where Probes 1-3 (the go/no-go tests for the Qwen-as-connectivity-verifier idea) were run, sections 9-14.
- `notebooks/e2e_harness/Pipeline3_R4_Validation_Benchmark_GPUOnly.ipynb` — **new this session.** Loads Qwen+v3-relation, validates candidate edges from the pre-built bundle, produces the pre/post 2×2.

**How the upgraded Pipeline 3 benchmark works end-to-end (the exact sequence this session ran):**
1. Real GPT-5.5-low extraction on 3 AG/RIVE sheets (`run_real_extraction_partB.py`, CPU, real OpenAI API cost, ~$1.72 total) → entities with `id`/`text`/`type`/`bbox_px`.
2. `relationship_pipeline.run_relationship_pipeline(entities, pdf_path, render_dpi, upgraded=True/False)` on each sheet, both configs → candidate relation sets.
3. `build_r4_bundle.py` (CPU) renders one crop per unique candidate edge (v3-relation adapter's trained crop format), zips, pushes to HF as `benchmarks/r4_bundle_2026-07-25.zip`.
4. GPU notebook pulls the bundle, Qwen+v3-relation validates each candidate (keep/reject) → post-LLM sets.
5. Opus (this session, manually) renders annotated overlays for the delta edges + hand-traces ground truth from the raw PDF → recall/precision verdict.

**Commands that matter:**
- `.venv-e2e/bin/python <script>` for all local/CPU work (never bare `python3` — the venv has the needed deps).
- No single "run everything" command exists — every script in this session was a bespoke one-off in the session's scratchpad (see the critical warning in Section 5 about where these live).

**External dependencies:** OpenAI API (`gpt-5.5` model id + `reasoning={"effort":"low"}` — note `"gpt-5.5-low"` as one string is NOT a real model id and 400s), HF Hub (dataset repo `timthy45/pnid-extraction-datasets`, model repo `timthy45/qwen3vl-pnid-domain-base`), the real `pnid_pipeline` source at `/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-extraction-agent/` (a sibling repo, imported directly via `sys.path.insert`).

---

### 4. RECENT WORK — WHAT JUST HAPPENED (HIGH PRIORITY)

**Chronological summary of this session:**

1. **Resumed from a prior session's Part B work** (2026-07-23): an "agreement-diff" comparing GPT-5.5-low's real connectivity claims against PDF vector-geometry tracing on 3 real AG/RIVE sheets had found near-zero agreement (3/31) despite a manual (Fable-model) adjudication showing GPT-5.5 was actually ~80% accurate — the low score was a measurement-category artifact (most true claims reference off-page equipment no single-sheet tracer can ever confirm).

2. **Ran 3 go/no-go probes** for a proposed "hybrid architecture" (geometry tracer proposes edges, a VLM verifies each with a bounded yes/no question):
   - Probe 1 (symbol-extent reconstruction from vector geometry): PASS.
   - Probe 2 (Qwen3-VL-8B bounded connectivity verification, 19 real crops, 80% bar): **FAIL, 52.6%** — confirmed via 5 separate test variants (forced-answer flatline, chain-of-thought, token-budget fixes, the actual fine-tuned v3-relation adapter in both its own prompt format and CoT) that this is a genuine capability gap, not an artifact. Qwen's failure mode: confident, plausible-sounding hallucinated traces.
   - Probe 3 (Qwen reading off-page connector text, 8 crops, 80% bar): **PASS, 87.5%.**
   - A GPT-5.5-low ceiling test (real API cost) on the same 19 pairs: 68.4-73.7%, still below bar — until an eyeball audit found the probe's OWN boxes were drawn on tag TEXT instead of the real symbol shape (same bug as the original MBD-0100 finding). Correcting 3 boxes + dropping 1 mislabeled pair moved GPT-5.5 to **83.3%, clearing the bar.**

3. **Proposed and built "Tier 1"** — 4 deterministic (model-free) upgrades to Pipeline 3, decided WHY: item #2 (backbone pass) and #4 (line-label filter) attack measured weak strata directly; item #1 (partitioning) makes an existing silent data-loss visible; item #3 (symbol-extent) was the most valuable but hit the wall described in Section 2.

4. **The user asked a sharp architectural question mid-session:** "doesn't the entity-extraction stage's Molmo2 pointing already solve the symbol-location problem?" **I initially answered this WRONG** (claimed Fable had never actually been used this session, based on a bad inference from `/model` command wording) and separately **initially claimed the shape-vs-label problem was already solved by Molmo2**, before the user corrected me and I found the real answer: the multi-arm entity-extraction design (`E2E_Harness_Plan.md`'s D-M1 decision) DOES solve this via Molmo2 + nearest-OCR-word pairing, but that arm was **never wired into** the relationship-stage testing this session — the relationship tests used prod's OCR/Vision-based tag extraction, which only carries text-label locations, not symbol shapes. **This is a real, unresolved architectural gap, not yet closed by code.**

5. **Built and validated the backbone pass (#2) against real data before trusting it:** checked all 762 local PID2Graph sheets — 84.6-84.8% of "walk through one inline valve/instrument" candidate pairs are already genuine direct GT edges (not invented), ~15% would be false positives. Real single-sheet benchmark: F1 0.104→0.156. **Then ran the full 12-sheet real OPEN100 aggregate** (the honest multi-sheet number, Dataset PID abandoned per Section 2): **F1 0.214→0.225** — smaller and messier than the single-sheet number suggested; recall +16%, precision −16%, fp grew more than the corpus check predicted (54% vs ~15%) because the corpus check only validated the single-fitting-hop case, not the longer chains the pass also produces in practice.

6. **User asked to "wire all upgrades into the pipeline and benchmark properly"** and specifically wanted BOTH pre-LLM and post-LLM results captured (explicit user design decision: "insace its giving worse results we have the result even without verification to show"). Built `relationship_pipeline.py`, ran it on all 3 real sheets in both configs (pre-LLM candidate counts — see artifact for exact numbers), built the R4 GPU notebook, ran the smoke gate: **R4 (local Qwen validator) kept 0/8 on every sheet — confirmed degenerate, matches Probe 2 exactly.**

7. **Decision, explicitly reasoned and stated to the user:** do NOT run the full 492-candidate R4 validation — it would almost certainly just reconfirm the same zero result, not worth the GPU time. This was proposed by me and not explicitly re-confirmed by the user before I moved on — **flag this as a decision the next AI should surface, not silently continue past.**

8. **Model/effort switch mid-session:** user switched to Opus 4.8 (high effort) specifically for the visual-adjudication work, on my recommendation (reasoning given: visual judgment quality matters more than reasoning depth for this bounded task).

9. **Opus (this session) performed the delta adjudication + recall verification** on PX-2368 only: rendered annotated overlays for the backbone-pass's added edges and the line-filter's removed edges, judged each against the real drawing; separately hand-traced ~26 real connections across the sheet's 3 main equipment clusters and scored the pipeline's recall. **Results: backbone-added edges were 0-1/5 correct** (4/5 falsely link two separate off-page connector labels — a NEW failure mode, not seen in the earlier corpus check); **line-removed edges were correctly spurious** (real precision win); **recall = 15/26 (58%) strict, 20/26 (77%) crediting near-misses, IDENTICAL in both configs** (the upgrades don't touch recall, only false-positive composition).

10. **Added all of this to the published artifact** (URL in Section 2) as a new dated section. **Did NOT add it to `Benchmark_Gaps_Register.md`** — offered, not confirmed/done.

11. **This turn:** pushed the 3 real extraction-result JSONs to HF (`benchmarks/extraction_2026-07-24/*.json` on `timthy45/pnid-extraction-datasets`) because they existed only in this session's ephemeral scratchpad and would otherwise be unrecoverable by a fresh session.

**What was discussed but NOT yet implemented:**
- A GPT-5.5-low R4 arm (cheap, no GPU, would answer "does a competent validator help" as a genuine architecture question, separate from the local-substitution question).
- Wiring the backbone pass into `agreement_diff.py`'s vector-tracer call (currently the AG/RIVE agreement-diff path does NOT include the backbone pass at all — only PID2Graph testing does).
- Tightening the backbone pass to single-fitting hops only (my recommendation after seeing the false-positive growth exceed the validated case) — proposed, not built.
- Recall-tracing the other 2 of 3 real sheets.

**Open threads / unresolved questions:**
- Is the "0 off-page claims" result on the fresh 2026-07-24 extraction real, or an artifact? (Investigated: the off-page equipment mentions this run got assigned narrow 110×20px label-shaped bboxes instead of falling back to raw text — same underlying shape-vs-label problem as blocked item #3, just showing up in a second code path. Not fully resolved.)
- Symbol-extent resolution (#3) needs a real design decision from Tom before any further engineering time goes into it.
- Whether to pursue wiring the multi-arm entity-extraction's Molmo2+tag_matching design into the relationship-stage tests (see point 4 above) — raised, not decided.

**Promote to project memory (durable, non-obvious lessons):**
- **The tag-text-bbox vs symbol-shape-bbox distinction is the single most recurring bug source in this entire project.** It caused: the original MBD-0100 near-zero-agreement mystery, the Probe 2 crop-boxing bug, and (likely) the "0 off-page claims" anomaly in the 07-24 rerun. Any future work touching entity bboxes for connectivity/tracing purposes must ask "is this a text-label location or a drawn-shape location" explicitly.
- **`/model` switches only take effect for the NEW session by default ("saved as your default for new sessions")** — but real usage data can show a model WAS active mid-session despite that wording (verified via `/usage` screenshot showing real `claude-fable-5` token spend this session). Do not assume the wording implies the switch failed — check actual usage if it matters.
- **GPT-5.5-low API calls this session used the real production model shape**: base model `"gpt-5.5"` + separate `reasoning={"effort":"low"}` field — `"gpt-5.5-low"` as one string 400s.
- **Real API/GPU spend this session:** ~$1.72 (3 sheets × GPT-5.5-low extraction, 2026-07-24 rerun) + ~$0.02 (19-pair ceiling test) — all explicitly authorized by Tom ("i dont mind spending that cash - spend it").

---

### 5. WHAT COULD GO WRONG

- **CRITICAL: most of this session's raw artifacts live ONLY in this session's ephemeral scratchpad** (`/private/tmp/claude-501/-Users-tomgeorge-pid-ml/10ffbddb-0f7d-4f19-a544-f1152513500c/scratchpad/`), which is **tied to this specific session ID and will NOT be accessible from a fresh conversation.** What IS recoverable (pushed to HF, durable):
  - `benchmarks/probe_bundle_2026-07-24.zip` (Probe 2/3 crops + answer keys)
  - `benchmarks/r4_bundle_2026-07-25.zip` (492 candidate-edge crops + manifest)
  - `benchmarks/extraction_2026-07-24/*.json` (3 real sheets' GPT-5.5 extraction — **just pushed this turn**)
  - `sheets/AG_PNID.zip`, `sheets/RIVE_LTTS_Sample.zip` (the source PDFs, pushed in an earlier session)
  - What is **NOT recoverable** and would need to be regenerated: the adjudicate-crops overlay renders (PX-2368 delta-pair PNGs), all the one-off analysis scripts (`build_r4_bundle.py`, `adjudicate_crops.py`, `run_backbone_open100.py`, etc. — none committed to git, none pushed anywhere). The PID2Graph GT corpus itself (762 sheets) also lives in a DIFFERENT prior session's scratchpad path (`.../a852824c-.../scratchpad/molmo_ft_prep_v2/pid2graph/`) — still on this same Mac's disk, so locally accessible, but not portable.
- **The `benchmarks/r4_validation_results_2026-07-25.json` push (notebook cell 7) was never confirmed to have run** — the user only pasted cell 6's console output (the `kept 0/8` lines). Verify this file exists on HF before assuming it does.
- **Assumption to verify, not trust:** whether Tom wants the full 492-candidate R4 run done anyway (I recommended against it, reasoning given in point 7 above, but this was my call, not an explicit "yes skip it" from Tom).
- **Assumption to verify:** the `PASSTHROUGH_TAG_TYPES` mapping in `relationship_pipeline.py` (`{valve, instrument, fitting, safety_device}`) is FROZEN but **not yet reviewed by Tom** — same pattern as `type_vocab.py`'s existing frozen-but-flagged mappings. Don't treat it as settled.
- **Edge case already found and worth remembering:** off-page connector-box text sometimes resolves to a real tag id with a narrow, label-shaped bbox rather than falling back to raw text — code that assumes "resolved to a tag id" means "genuinely on-sheet, real symbol" will be wrong.
- **Technical debt:** none of this session's code has any automated test suite. All validation was ad hoc scripts + manual/visual inspection. If this code needs to be trusted long-term, it needs real tests.

---

### 6. HOW TO THINK ABOUT THIS PROJECT

1. **Core design philosophy:** never trust a single signal. Every result in this session was cross-checked against an independent second signal — geometry vs LLM claims, corpus-level validation before a targeted benchmark, a "third-signal" model (Fable, then Opus) adjudicating by eye when no ground truth exists. This was chosen because the project has been burned before by trusting a metric that turned out to measure the wrong thing (the original MBD-0100 near-zero-agreement mystery, the graph-encoding mismatch that zeroed every relation score in an earlier session).
2. **Most common mistake a new person would make:** assuming a tag's `bbox_px` is the location of the drawn symbol it names. It is almost always the location of the printed TEXT LABEL, which can be meaningfully far from the actual shape. This single confusion has caused multiple real bugs across multiple sessions.
3. **What looks like it should be refactored but intentionally should NOT be:** the fact that Tier-1 upgrades #1/#2/#4 were tested on DIFFERENT datasets with DIFFERENT tracers (PID2Graph+raster for #2, AG/RIVE+vector for #1/#4) looks like an inconsistency worth "fixing" into one unified test. It is not a bug — each upgrade is only *scoreable* where its precondition exists (real edge GT only exists on PID2Graph; real LLM claims only exist on the AG/RIVE run). Unifying them would mean losing the ability to score at all. This was an explicit, reasoned design choice this session, not an oversight.

---

### 7. DO NOT TOUCH LIST

- Do NOT commit anything in this repo unless Tom explicitly asks — standing rule, confirmed multiple times this session.
- Do NOT run the full 492-candidate R4 Qwen validation without checking with Tom first — my recommendation was to skip it as low-value; that recommendation itself should be revisited with Tom, not silently acted on either way.
- Do NOT treat the "recall %" numbers (58%/77%) as certified ground truth — they are an Opus third-signal trace on ONE of three sheets, scoped to 3 equipment clusters, with off-page connections explicitly excluded. Do not extrapolate them to the other 2 sheets without doing the same tracing work.
- Do NOT assume PID2Graph's Dataset PID tree can be used for anything requiring R1 (raster tracer) without first solving the resolution/scaling problem (Section 2) — it will not finish in reasonable time as-is.
- Preserve the pre-LLM/post-LLM split pattern in any future R4-style validator test — this was an explicit Tom design requirement ("insace its giving worse results we have the result even without verification to show"), not an implementation detail to simplify away.
- Ask before wiring the backbone pass into `agreement_diff.py` (the AG/RIVE path) — it currently only runs in the PID2Graph path; extending it changes what the AG/RIVE agreement numbers mean.
- Ask before spending further real API/GPU budget (the ~$1.72 + $0.02 this session was explicitly pre-authorized; further spend should get the same explicit sign-off Tom already established as his preferred pattern).

---

### 8. CONFIDENCE & FRESHNESS

- Section 1 (Project Identity): ✅ HIGH — read directly from `CLAUDE.md` this turn, except the "beyond stated scope" observation which is my own inference — ⚠️ MEDIUM on that specific point.
- Section 2 (What exists): ✅ HIGH — `git status`/`git log`/`ls` run this turn, HF push just performed and confirmed.
- Section 3 (Architecture): ✅ HIGH for file contents (read directly this session) — ⚠️ MEDIUM for "how the 3 pipelines work end-to-end," which is carried forward from the artifact's own description, not independently re-verified against the real `pnid-intelligence-agent`/`pnid-extraction-agent` source in this turn.
- Section 4 (Recent work): ✅ HIGH — this is a direct account of this session's own actions and real tool outputs.
- Section 5 (Risks): ✅ HIGH for the scratchpad-portability finding (verified by direct `ls` this turn) — ❓ LOW for whether the R4 notebook's cell 7 actually ran (genuinely unverified, flagged explicitly above).
- Section 6 (How to think about it): ⚠️ MEDIUM — my own synthesis/judgment, not an independently-verifiable fact.
