# Extraction-Agent Local-Model Benchmark — Execution Plan

**Status:** plan, not yet executed. Written 2026-07-17 for execution by Sonnet 5.
**Goal:** run the REAL `pnid-extraction-agent` pipeline (unmodified where possible) with
local models swapped into its LLM slots — Qwen3-VL-8B for all reading, PaddleOCR for OCR,
plus a NEW Molmo2+OCR supplementary-detection slot — and score it with the SAME `revR`
metric, on the SAME 13 real sheets, against the SAME `reviewed_truth.json` ground truth
that produced the recorded GPT-5.5 baselines. Apples-to-apples on the pipeline; only the
models change.

**Decision this feeds:** if the local stack lands at/near GPT-5.5-low's recorded number,
the substitution story is proven on the extraction agent and we move to the intelligence
agent next (user's stated sequencing).

**REVISED SCOPE (2026-07-17, after Phase 1 discovery — see §11):** the recorded GPT-5.5
baselines (0.836/0.813) almost certainly ran through `ocr_reasoning_extract`
(`pnid_pipeline/ocr_reasoning.py`), the real default pipeline mode (`config.yaml`:
`mode: ocr_reasoning`) — NOT the `read_shapes`/`read_regions` CV-hybrid path this plan
was originally architected around (confirmed by the recorded per-sheet durations:
900–1,600+ seconds/sheet, only explicable by a few large `max_tokens=32000` reasoning
calls, not many small parallelized 40-token crop reads). Per explicit user decision:
**build and run BOTH paths, report all three** — the real GPT-5.5 baseline, the local
stack through `ocr_reasoning` (the true apples-to-apples comparison), and the local
stack through the CV-hybrid path (a bonus second architecture, not directly comparable
to the recorded numbers but interesting in its own right). §3 below now has TWO
architecture subsections; §7's run protocol reports 5 numbers total (1 recorded + 2
local configs × with/without Molmo2).

---

## 1. Comparability contract (what makes this apples-to-apples)

Same, unchanged:
- **Pipeline code:** `pnid_pipeline/` (`extract_page` entrypoint) — classify/triage, CV
  shape detection, grounded reads, snap, assertions, adjudication passes, hierarchy,
  reconcile, assemble. Run via its real code path, not a reimplementation.
- **Prompts:** the pipeline's own `_SHAPE_PROMPT` / region / adjudicator prompts, as-is.
  (One permitted deviation class: FORMAT adaptation for models without native
  structured-output support — same precedent as Arm P/L work. Content stays identical.)
- **Ground truth:** `scripts/eval/review_reads/<stem>/reviewed_truth.json` (13 sheets we
  have PDFs for; `Sample_PID` excluded — no PDF in our packages).
- **Metric:** `revR` computed by THEIR `score.py` functions (`review_keep`,
  `load_reviewed_truth`, `review_recall`) — import/reuse, don't reimplement. (Our
  `src/e2e_harness/score_revR_real_sheets.py` already has a verified verbatim copy if
  import is awkward in Colab.)
- **Mode:** whatever `PNID_MODE` the recorded GPT runs used (Phase 0 confirms; expected
  default "cv" route via `_alt_bench.py`).

What changes: `call_llm` (the injected model callable), the OCR engine inside
`tiled_ocr_words`, and ONE addition — the Molmo2 candidate source (run both with and
without it; see ablation).

## 2. Recorded baselines (real numbers from `history/model_comparison_1782539273.json`)

| Model | mean_revR (14 sheets) | $/drawing | sec/drawing |
|---|---|---|---|
| GPT-5.5 high (`OPENAI_REASONING_EFFORT=high`, the proxy default) | **0.836** | 4.21 | 982 |
| GPT-5.5 low | **0.813** | 0.96 | 295 |
| Sonnet 4.6 | 0.811 | 0.62 | 201 |
| Gemini 3.1 Pro | 0.752 | 0.78 | 227 |

Success bars (proposed, POC framing): **≥ 0.75 mean_revR = credible POC** (beats Gemini,
within ~8% of GPT-5.5-low); **≥ 0.813 = parity headline**. Report per-sheet always.
Do NOT compare against our Arm-P revR partial run (0.697/3 sheets) — different pipeline
(intelligence agent), explicitly not comparable.

## 3. Architecture — TWO local configs, run in parallel

Both share: PaddleOCR swap, `build_qwen_call_llm` shim (§3.3), same 13 sheets, same
`revR` scoring. They differ in WHICH real pipeline code path they run through.

### 3.A `L-ocr` — the `ocr_reasoning` path (TRUE apples-to-apples vs. recorded baselines)

| Slot | Today (prod) | Local swap |
|---|---|---|
| `_ocr_words`/`tiled_ocr_words` (called from inside `ocr_reasoning.py`, imported into ITS OWN namespace — patching `vision.py` alone does NOT reach it, confirmed Phase 1) | Google Cloud Vision, tiled+auto-upscale | **PaddleOCR** — patch `pnid_pipeline.ocr_reasoning.tiled_ocr_words` directly, not just `vision.py`'s copy |
| Main reasoning call (`_prompt`, `OcrResult`/`OcrTag` schema, up to `max_tokens=32000`, chunked via `_split_regions` on dense sheets) | GPT-5.5 via `build_openai_call_llm` | **Qwen3-VL-8B base** via `build_qwen_call_llm` (same shim, same interface — `call_llm(prompt, schema_model, images=[overview_b64], model=, max_tokens=)`) |
| Recovery pass (`_recovery_prompt`, same schema, re-asks about unclaimed OCR tokens) | Same model | Same Qwen3-VL-8B callable |
| **NEW: Molmo2 slot, redesigned for this path** | — | See below — feeds as SYNTHETIC OCR TOKENS, not a `snap_candidates` merge (that merge doesn't exist in this path) |

**Molmo2 integration for `L-ocr` (the correct redesign, not the original §3.1 idea):**
This path never does per-shape crop reads — it reasons over one flat list of
`(id, text, cx, cy)` OCR tokens + one overview image. Molmo2's role here is to catch
symbols/bubbles that OCR's text-detection missed ENTIRELY (no token at all for that
bubble) — the same gap the pipeline's own comment flags ("Use the image to catch tags
OCR split/missed"). Concretely:
1. Run Molmo2 per-class pointing (same validated 512px/×2 config) over the rendered
   page, same as before.
2. For each Molmo2 point, check if an OCR token already exists within a small radius
   (~30px) — if yes, skip it (OCR already covers this location, adding a duplicate
   synthetic token would just create noise/double-votes).
3. For points with NO nearby OCR token: pair with the SAME nearest-OCR-word logic
   (within the larger 120px radius) to borrow a text value, OR if truly no text is
   nearby, emit the token with placeholder text `"?"` + a distinct `source` marker so
   the reasoning prompt can still be told "there's a symbol here, but no legible tag
   text was found" — let the LLM decide whether to include it (matches the pipeline's
   own "identify what's real" philosophy, doesn't fabricate a tag).
4. Inject these as EXTRA rows appended to `idx_words` (the same list `_prompt`/
   `_recovery_prompt` render into the token list), with `id`s continuing past the real
   OCR token count so `word_ids` citations stay unambiguous. Tag these injected rows
   with a documented in-code comment (`# molmo_synthetic`) — no schema change needed,
   `OcrTag.word_ids` already accepts any int.
5. Attribution: after scoring, separately compute revR using ONLY the real-OCR-token
   run vs. the run with Molmo tokens added, to isolate the slot's real contribution
   (same ablation spirit as the original plan, adapted to this path's actual mechanics).

### 3.B `L-cv` — the CV-hybrid path (`read_shapes`/`read_regions`, Phase 1's original build)

Unchanged from the original plan — this is what Phase 1 already built and
fake-tested successfully:

| Slot | Today (prod) | Local swap |
|---|---|---|
| `tiled_ocr_words` (vision.py) | Google Cloud Vision | **PaddleOCR**, patch `vision.py` (this path DOES read from there directly — confirmed working in Phase 1's acceptance test) |
| `read_shapes` (per-symbol crop, `_OneTag`, 40 tok) | GPT-5.5 | Qwen3-VL-8B via `build_qwen_call_llm` |
| `read_regions` (text-cluster reads, `_ManyTags`, cap=250) | GPT-5.5 | Qwen3-VL-8B, same callable |
| Adjudicator (re-read unclaimed, ≤2 passes) | `PNID_ADJUDICATOR` model | Qwen3-VL-8B, same callable |
| **Molmo2 slot (original §3.1 design, NOT yet wired per Phase 1 build log)** | — | 4th candidate source merged via `snap_candidates(shape_cands + region_cands + ocr_cands + molmo_cands, symbols, R)` in `extract.py` — requires a small, explicit one-line addition to that call (Phase 1 found no existing extension point; this is a deliberate, minimal, documented source change, not a rewrite) |

Molmo2 candidate construction (unchanged from original plan): 512px/×2 tiles,
per-class pointing, parse via `e2e_bench/backends/parse_molmo.py`, dedupe across tile
overlaps, pair with nearest OCR word within 120px (no word in radius → synthetic box
appended to `symbols` for adjudication, no candidate emitted), dict shape
`{"text":.., "raw":.., "box":(x0,y0,x1,y1), "source":"molmo_point", "shape":"circle",
"signals":["molmo_point"]}` (confirmed passes through `assemble.py` unfiltered, Phase 0
finding #4).

To run `L-cv` under the plan's `PNID_MODE=cv` override (a deliberate, documented
deviation from the recorded baselines' actual mode — see §11 discovery), since only
`ocr_reasoning`(`L-ocr`) matches the true recorded default.

### 3.3 `build_qwen_call_llm` — the local callable (shared by BOTH configs)

Contract (verified from `grounded_read.py` + `openai_proxy.py`):
`await call_llm(prompt, schema_model, images=[b64,...], model=..., max_tokens=...)`
→ returns a validated instance of `schema_model` (pydantic), and the callable object
carries a `usage` dict (`_usage_snapshot` reads it — keep the same fields:
in/cache_w/cache_r/out/calls per model name, zeros are fine for cache).

Implementation: same technique as `src/e2e_bench/assembly/local_qwen_client.py` —
render the pydantic schema into a fenced-JSON instruction appended to the prompt,
`apply_chat_template` with decoded image(s), `generate` (greedy), extract JSON via
`e2e_bench/backends/parse_json_common._extract_json_text`, `schema_model.model_validate`.
On parse failure: return an empty/default instance (e.g. `_OneTag(tag="")`) — the
pipeline already treats empty reads as "no tag", which is the honest failure mode.
Async: pipeline awaits with `Semaphore(6)` concurrency, but local GPU generation is
serial — implement as an async wrapper around a lock-guarded sync generate (concurrency
collapses to 1; that's fine, note it in the latency report).

## 4. Phase 0 — investigations (local Mac, read-only, ~1–2h)

Each item: read the file, write the answer into `Extraction_Agent_Local_Plan.md` §11
(build log) before coding.

1. `_SHAPE_PROMPT`, `_OneTag`, region prompt/schema, adjudicator prompt: exact text and
   pydantic shapes (`grounded_read.py`). Needed to build the Qwen JSON instruction.
2. How the recorded GPT runs were configured: `_alt_bench.py` + `bench_summary.py` —
   `PNID_MODE`? route forced? `max_vlm_reads`? fewshot? Replicate exactly.
3. Signals/tier logic downstream (`validate/assemble/reconcile`): what does a candidate
   need to survive to the final Tag list — does a `molmo_point` signal (unknown string)
   pass through or get filtered? Where does `cv_shape` matter for tiering?
4. `call_llm` full contract: read `llm_proxy.py` too (the Anthropic path) — confirm the
   `schema_model` handling (tool-call vs JSON-mode?) so the Qwen shim matches semantics,
   and confirm the usage-dict attribute name `_usage_snapshot` expects.
5. `triage_page` / `classify.py`: any LLM usage? (expected: none for route decision —
   confirm.) Hierarchy/reconcile: any LLM? (expected none.)
6. `read_regions` internals: prompt, schema (multi-tag?), `cap=250` — anything needing
   different JSON handling than `_OneTag`.
7. Confirm per-sheet expected call volume: `max_vlm_reads=600` cap × 13 sheets worst
   case + regions (≤250) + adjudication — estimate GPU-hours at measured tokens/call
   (smoke test gives the real number; abort criteria in §7).

## 5. Phase 1 — local builds + tests (Mac, CPU, no GPU needed)

1. **PaddleOCR adapter** (`paddle_ocr_words(img, *, tile=1400, overlap=220)`): reuse
   vision.py's tiling loop verbatim, swap `gv_words` for PaddleOCR per-tile inference,
   return `Word` tuples. Wire-in: monkeypatch `pnid_pipeline.vision.tiled_ocr_words` (and
   the preflight `vision_key` check) from the harness — do NOT edit agent source.
   Test locally on one rendered sheet: word count sanity vs our earlier PaddleOCR runs.
2. **`build_qwen_call_llm(generate_fn)`** with injected `generate_fn` (same pattern as
   `local_qwen_client.py`) → unit-test on Mac with a fake generate_fn: `_OneTag` success,
   garbage output → empty tag, multi-image call, usage-dict shape.
3. **Molmo2 candidate mapper** (`molmo_candidates(points_by_class, ocr_words, R)`):
   pure function, point→word pairing + candidate dicts + synthetic symbol boxes.
   Unit-test with synthetic points/words incl. no-word-in-radius and tile-overlap dedupe.
4. **Harness runner** (`run_extraction_local.py`): renders PDF (150dpi, same as our revR
   runner), monkeypatches OCR + preflight, injects call_llm, optionally injects
   molmo_cands (flag), calls the REAL `extract_page`, writes preds JSON in THEIR format
   (match `run_eval.py`'s `preds/<stem>_p1.json` shape so THEIR score path works), scores
   revR. Test end-to-end on Mac with BOTH fakes (fake generate_fn returning canned tags,
   fake Molmo points) on 1 sheet — proves the whole integration before any GPU spend.

Acceptance for Phase 1 (original 4 items, `L-cv` config): fake-backed end-to-end run on
`GD-T-435-DT-2042-056` produces a preds JSON their `score.py` scores without error. —
**DONE, see §11.**

**Phase 1b — added for `L-ocr` (the true apples-to-apples config), not yet built:**

5. **Fix the OCR monkeypatch for `ocr_reasoning` mode**: patch
   `pnid_pipeline.ocr_reasoning.tiled_ocr_words` directly (not just `vision.py`'s copy —
   `ocr_reasoning.py` imported the name into its own namespace at module load, so patching
   the source module afterward doesn't reach it — this is the exact bug Phase 1 already
   hit and is documented in §11).
6. **`molmo_synthetic_tokens(points_by_class, ocr_words, near_radius=30, pair_radius=120)`**
   — pure function per §3.A's redesign: skip points already covered by a nearby real OCR
   token (dedup vs. duplicate signal), else pair with nearest word within `pair_radius` or
   emit placeholder text `"?"`, return rows shaped like `idx_words`
   (`(id, text, cx_norm, cy_norm)`) with ids continuing past the real token count. Unit-test
   with synthetic points/words: covered-point skip, paired point, placeholder-text point.
7. **Extend `run_extraction_local.py`** to support a `pipeline_mode` parameter
   (`"cv"` → `L-cv` as already built; `"ocr_reasoning"` → `L-ocr`, patches
   `ocr_reasoning.tiled_ocr_words`, injects synthetic tokens into the `words` param
   `ocr_reasoning_extract` already accepts — see its signature, it takes an optional
   `words` override precisely for this kind of injection).
8. Acceptance for Phase 1b: same fake-backed end-to-end test, but with
   `pipeline_mode="ocr_reasoning"` AND `PNID_MODE` left at its real default (do NOT
   override to `cv`) — proves `L-ocr` runs through the actual default path with no
   env-var deviation.

## 6. Phase 2 — Colab notebook (`notebooks/e2e_harness/ExtractionAgent_Local_GPUOnly.ipynb`)

- Same conventions as the ArmL notebooks (HF-only transfer, no Drive, token in config
  cell, cascade-mode H6 exception note).
- **Packaging:** extend `scripts/package_agent_src_for_colab.sh` to include
  `agents/pnid-extraction-agent` (+ its `scripts/eval/review_reads` ground truth and
  `score.py`) in the zip. ⚠️ **USER CHECKPOINT:** the 13 sheet PDFs must also reach
  Colab. They are marked Restricted/EAR99 — get Tom's explicit OK before uploading them
  to the private HF repo (same repo as agent source, private=True). Do not proceed
  without it.
- **Two-phase-per-corpus model schedule** (memory-safe on a 24GB GPU, avoids co-residency):
  - Phase A: load Molmo2 → for ALL 13 sheets: render, PaddleOCR (CPU), tile, per-class
    pointing → cache `molmo_points/<stem>.json` + `ocr_words/<stem>.json` to disk → free
    Molmo2.
  - Phase B: load Qwen3-VL-8B (base, bf16 — exact recipe from the skid-matrix notebook)
    → for ALL 13 sheets, run BOTH configs off the same cached data: `L-ocr` (real default
    mode, no `PNID_MODE` override, synthetic-token Molmo injection) and `L-cv`
    (`PNID_MODE=cv`, `snap_candidates` Molmo merge) → 4 preds sets total per sheet
    (`L-ocr`, `L-ocr+M`, `L-cv`, `L-cv+M`) → score all 4.
- Per-sheet wall-clock + generation-count printed; abort thresholds from §7.

## 7. Phase 3 — run protocol (discipline)

1. **Smoke (1 sheet):** `PX-2368-0180004-001` (n=50, the human-curated one, GPT-5.5
   scored 0.98 revR here — maximal signal). Gates: parse-failure rate < 20% on `_OneTag`
   reads; ≥ some candidates from every source; wall-clock ≤ ~25 min/sheet on the GPU in
   use. Fail a gate → stop, report, fix before spending more.
2. **Dev tuning (3 sheets, ONLY these):** `PX-2368-0180004-001`, `GD-T-435-DR-2031-030`,
   `PX-2365-0150022-001` (small/medium, both families). Tunables: Molmo pairing radius,
   Molmo class list, Qwen max_new_tokens, nothing else. NOTE for honesty: the GPT
   baseline itself was iterated on this full corpus (11 recorded tuning runs), so
   dev-set tuning is not an unfair advantage — but still report dev vs held-out split
   separately.
3. **Full run (all 13):** FOUR local configurations, per user decision to test both
   architectures —
   - **L-ocr** (Qwen + PaddleOCR through the real default `ocr_reasoning` path, NO Molmo) —
     the TRUE apples-to-apples vs. recorded GPT-5.5 (0.836/0.813).
   - **L-ocr+M** (same + Molmo2 synthetic-token injection) — isolates the new slot on
     the correct-path config.
   - **L-cv** (Qwen + PaddleOCR through the CV-hybrid path, `PNID_MODE=cv`, NO Molmo) —
     bonus second architecture, not directly comparable to recorded baselines.
   - **L-cv+M** (same + Molmo2 via `snap_candidates` merge) — isolates the slot on the
     bonus path.
4. Molmo attribution: for both `+M` configs, count truth tags hit ONLY via
   Molmo-sourced candidates/tokens → the slot's direct revR contribution, reported
   explicitly per config (the mechanism differs between L-ocr+M and L-cv+M, so their
   attribution numbers are not directly comparable to each other, only within each).

## 8. Phase 4 — scoring + reporting

- **Headline 3-way comparison** (what the user asked for): GPT-5.5 recorded baseline vs.
  L-ocr (+M) vs. L-cv (+M) — mean revR, clearly labeled which local config is the real
  apples-to-apples one (L-ocr) and which is the bonus architecture test (L-cv).
- Per-sheet table: revR for all of {GPT-5.5-high, GPT-5.5-low (history), L-ocr, L-ocr+M,
  L-cv, L-cv+M}, parse-failure rate, sec/sheet, generations/sheet.
- Means over: all 13; held-out 10 only (dev excluded); AG vs PX family split (we saw a
  family gap on the intelligence-agent side — check if it exists here too).
- `results.csv`: one row per configuration (schema per CLAUDE.md; `stage` =
  `extraction_agent_e2e(revR,13 sheets)`), 4 new rows (L-ocr, L-ocr+M, L-cv, L-cv+M).
- Update `Extraction_Agent_Local_Plan.md` §11 build log with every real discovery
  (same practice as Conversion_Layer_Plan.md).
- No HTML artifact until Tom asks.

## 9. Risks / honest unknowns (state these in the final report too)

1. **Qwen JSON reliability under the pipeline's own prompts** — untested. Mitigated by
   per-call parse-failure tracking + the empty-tag fallback; smoke gate at 20%.
2. **Molmo2 transfer to these sheets** — its 0.628 F1 was on Gupta-style data at
   512/×2; these are denser industrial sheets. The slot may add noise; that's exactly
   what the ablation measures. A net-negative result is a valid, reportable finding.
3. **Serial local generation vs API concurrency** — sec/drawing will not be comparable
   to the recorded GPT timings in kind (GPU-seconds vs API-seconds); report both, don't
   blend.
4. **PaddleOCR vs Google Vision** — known engine downgrade; affects `locate()` box
   grounding and pairing, not just word lists. If revR lands low, an OCR-only ablation
   (rerun 2 sheets with a GOOGLE_CLOUD_VISION_API_KEY if Tom can supply one) isolates it.
5. **`_usage_snapshot` / preflight coupling** — monkeypatch points may be more numerous
   than §5 assumes; Phase 0 item 4 de-risks.
6. **GPU memory** — Qwen3-VL-8B bf16 on a 24GB card is tight but proven in this project;
   Molmo2 phase runs separately by design.

## 10. User checkpoints (Sonnet: STOP and ask at each)

1. Before uploading the 13 sheet PDFs to HF (Restricted/EAR99 marking) — §6.
2. After smoke sheet — show gates, get go/no-go for the full spend.
3. After full run — before any results.csv/report write-up, show raw numbers.

## 11. Build log — real discoveries (append during execution)

**Phase 0 complete (2026-07-17). Findings, all confirmed by reading real code:**

1. **No separate adjudicator prompt exists.** The adjudication pass reuses `_SHAPE_PROMPT`/
   `_OneTag` via `GR.read_shapes(img, unclaimed, words, call_llm, adjudicator)` — same
   function, same prompt, just a different `model` string. Nothing extra to build for
   the "adjudicator" slot — it's Qwen3-VL again, called through the same shim.
2. **`_alt_bench.py` is the exact script that produced the recorded GPT-5.5 baselines.**
   Key config, must replicate exactly: `force_route=None` (route is whatever
   `triage_page` naturally decides per sheet — do NOT force a route), `PNID_MODE` unset
   (defaults to `"cv"`, confirming the CV+grounded-VLM path, not agentic/ocr_reasoning),
   `PNID_ADJUDICATOR` set to the same model as `PNID_VISION_READER`, concurrency=3
   drawings in parallel (irrelevant for local serial GPU generation). The `CORPUS` list
   in this file maps every stem to `~/Downloads/AG_PNID/...` / `~/Downloads/RIVE/...` —
   confirms our sheet set and paths are exactly right.
3. **Routes are "B" or "A+B", never pure "A".** `triage_page` (`triage.py`) picks based
   on the PDF's embedded text layer quality: sparse/fragmented text → "B" (vision only);
   good text layer → "A+B" (Path A text-layer extraction runs in ADDITION to Path B, as
   corroboration — the code explicitly never routes to pure Path A alone, comment: "single-
   pass text-merge alone underperforms"). **Path A (`path_a.py`) is 100% deterministic
   (pdfplumber text-layer parsing, $0, no model call at all)** — it needs NO swap and
   runs identically regardless of which model powers Path B. This means per-sheet, our
   job is only to swap Path B's model calls; Path A (when triggered) contributes for
   free and is already apples-to-apples by construction.
4. **Signals pass through unregistered/unfiltered** — `assemble.py` does
   `signals=c.get("signals", [])` with no whitelist. A new `"molmo_point"` signal value
   needs no registration anywhere; it flows straight through to the final Tag list's
   `signals` field. One less thing to build.
5. **`classify.py` has no LLM call** — confirmed by grep, matches plan assumption.
6. **Config defaults confirmed** (`config.yaml`): `max_vlm_reads: 600`,
   `vision_reader: claude-sonnet-4-6` (repo default, overridden by `PNID_VISION_READER`
   at runtime — same pattern as `pnid-intelligence-agent`'s `agent.yaml`).
7. **Remaining Phase 0 items not yet checked:** full `llm_proxy.py` (Anthropic path) read
   for `_usage_snapshot` attribute contract — needed before finalizing `build_qwen_call_llm`'s
   usage-dict shape; `read_regions`' `_ManyTags`/cap=250 internals already read in full
   (§ item 6, see `grounded_read.py` above — no open questions there).

7. **`call_llm.usage` contract confirmed** (`llm_proxy.py`): must be a plain dict
   attribute, shape `{model_name: {"in": int, "cache_w": int, "cache_r": int,
   "out": int, "calls": int}}`. `build_qwen_call_llm`'s returned callable must expose
   this exact attribute/shape (real `calls` count; token fields can be 0 if not cheaply
   available locally — zeros are honest, not a fabrication, since local generation has
   no per-token API billing to report).

Phase 0 fully complete. **Next: Phase 1 (local CPU builds) — dispatched to a background
build agent with this full build log as context.**

**Phase 1 complete (2026-07-17). Built under `src/extraction_local/`:**

- `qwen_call_llm.py` — `build_qwen_call_llm(generate_fn)`. Reuses
  `e2e_bench/backends/parse_json_common._extract_json_text` verbatim for JSON extraction;
  falls back to a no-arg default instance (`schema_model()`) on any parse/validate failure.
  `.usage` dict shape confirmed exactly against `llm_proxy._record` (`in/cache_w/cache_r/out/calls`
  per model name). 5 unit tests, all passing (`_OneTag` success, garbage->empty tag, multi-image
  `_ManyTags` call, usage-dict shape, default-model-name path).
- `paddle_ocr.py` — `paddle_ocr_words(img, tile=1400, overlap=220)`. Same tiling stride math as
  `vision.tiled_ocr_words` verbatim. **Confirmed PaddleOCR 3.7.0's `.predict()` accepts an
  in-memory BGR numpy array directly** (not just a file path) — no per-tile disk round-trip
  needed, simpler than expected. Verified for real: 133 words on the smallest sheet
  (`GD-T-435-DT-2042-056`) at 150dpi.
- `molmo_candidates.py` — `molmo_candidates(points_by_class, ocr_words, radius=120)`, pure
  function. Candidate dict shape matches `grounded_read.Candidate` exactly. Dedup radius/2,
  keep-first-occurrence. No-word-in-radius still emits the synthetic ~2R box into
  `extra_symbol_boxes` but no candidate. 5 unit tests passing (hit, miss, dedup, nearest-wins,
  multi-class).
- `run_extraction_local.py` — harness: installs monkeypatches (`vision.tiled_ocr_words`,
  `vision.vision_key`, AND the same two names re-imported into `pnid_pipeline.extract`'s own
  namespace — see discovery below), builds `call_llm` via `build_qwen_call_llm`, calls the REAL
  `extract_page(path, 0, cfg, call_llm, force_route=None)`, writes
  `preds_local/<stem>_p1.json` in the exact `_alt_bench.py` shape
  (`result.model_dump(by_alias=True)`), scores with the REAL, imported (not copied)
  `scripts.eval.score.{load_reviewed_truth,review_keep,review_recall}`.

**Environment setup:** `pnid_pipeline` was NOT pip-installed — like `pnid_agent` in this same
venv, `pnid-extraction-agent` has no `pyproject.toml`/`setup.py` by design (its own
`requirements.txt` header explains: "loaded DYNAMICALLY... flat-module layout has no single
importable package"). Made importable in `.venv-e2e` the same way `pnid_agent` already is: a
one-line `.pth` file (`pnid_extraction_agent.pth`) pointing at the agent's repo root. Installed
one missing dependency: `pdfplumber` (needed by `path_a.py`, Path A text-layer extraction).

**Real discoveries / deviations from the plan, found by actually running this:**

1. **`config.yaml`'s `pipeline.mode` default is `ocr_reasoning`, NOT `cv`.** The Phase 0 build
   log's claim ("`PNID_MODE` unset defaults to `"cv"`") was wrong — checked via
   `git log -p -- config.yaml`: the file has had `mode: ocr_reasoning` since it was added in a
   single commit, never `cv`. This means the recorded GPT-5.5/Sonnet/Gemini baselines
   (`_alt_bench.py` never sets `PNID_MODE`) almost certainly ran under `ocr_reasoning`, not the
   `cv`+grounded-VLM path this whole plan's architecture (§3, `read_shapes`/`read_regions`, the
   Molmo slot in `_route_b_candidates`) is built against. `ocr_reasoning` mode uses a completely
   different schema (`OcrResult`/`OcrTag` in `ocr_reasoning.py`, not `_OneTag`/`_ManyTags`) and a
   different candidate-generation path entirely. **First real run under the true default hit this
   directly: `route` came back `"agentic"` (the `DrawingMeta.route` field is hardcoded to
   `"agentic"` for ALL THREE of agentic/agentic_tools/ocr_reasoning modes — a naming quirk, not a
   real route value) and `call_llm.usage` was EMPTY — the injected Qwen shim was never even
   called**, because `ocr_reasoning.py` imports `tiled_ocr_words` into its OWN module namespace
   (`from .vision import tiled_ocr_words`) — patching `pnid_pipeline.vision`/`extract`'s copies
   isn't enough for that mode; the REAL Google Vision path silently ran (and returned nothing
   useful with a dummy key).
   **Decision needed from Tom before Phase 2:** either (a) build `ocr_reasoning`-mode support too
   (new schema, new candidate flow, Molmo slot needs a different integration point), or (b)
   accept `PNID_MODE=cv` as a deliberate, documented deviation from how the baselines were
   produced (no longer a clean apples-to-apples "only the model changes" comparison), or
   (c) confirm with Tom/whoever ran `_alt_bench.py` whether `PNID_MODE=cv` was actually exported
   in their shell when the recorded baselines were produced (git history + code both say the
   in-repo default is `ocr_reasoning`, but an out-of-repo env var can't be verified from here).
2. **Phase 1's harness now forces `PNID_MODE=cv`** (env var, not `force_route` — `force_route`
   stays `None` per the plan's requirement) specifically so the four built components exercise
   the code path they were designed for. This let the acceptance test actually verify the
   injected `call_llm`/OCR swap end-to-end (usage `calls=11`, 22 tags produced, `route=B`).
   Also patched `ocr_reasoning.tiled_ocr_words` defensively for whichever mode decision Phase 2
   lands on.
3. **Molmo candidates are NOT yet wired into `_route_b_candidates`'s merge.** `run_one_sheet`
   accepts `molmo_points_by_class` as a parameter (interface completeness, per plan §5 item 4)
   but it's a documented no-op placeholder — there is no existing monkeypatch point inside
   `extract.py`'s `_path_b_candidates` for a 4th candidate source; wiring it in requires either a
   small, explicit agent-source change (adding one line to the `snap_candidates` call) or
   overriding the whole `_path_b_candidates` function. Out of scope for Phase 1 plumbing; flagged
   for Phase 2.

**Acceptance test: PASSED.** Fake `generate_fn` (canned `_OneTag`/`_ManyTags` JSON, detected by
prompt content) + real `extract_page(force_route=None)` on `GD-T-435-DT-2042-056`, `PNID_MODE=cv`:
no exceptions, `route=B`, 22 tags written to `preds_local/GD-T-435-DT-2042-056_p1.json` (exact
`_alt_bench.py`-format dump), `call_llm.usage = {'claude-sonnet-4-6': {..., 'calls': 11}}`
(model-name string flows through untouched — no real Claude call made), and the REAL, imported
`review_recall`/`load_reviewed_truth`/`review_keep` computed `revR = 0.636` (7/11 truth tags —
expected, since `TEST-101` is a fake constant tag, not a real read; this is a plumbing number, not
a quality number, exactly as designed). PaddleOCR word-count sanity check: 133 words at 150dpi on
the same sheet.

**Not done (explicitly out of Phase 1 scope per the task):** Molmo slot wiring into the real
candidate merge (item 3 above), any GPU/Colab work, any upload of the sheet PDFs anywhere.

**Phase 1b complete (2026-07-17). Built `src/extraction_local/molmo_synthetic_tokens.py` (new,
4 unit tests passing) + extended `run_extraction_local.py` with a `pipeline_mode` param
(`"cv"` unchanged / `"ocr_reasoning"` new).**

Real discoveries:
1. The `ocr_reasoning.tiled_ocr_words` monkeypatch (needed since that module imports the name
   into its own namespace) was ALREADY correctly installed by Phase 1's
   `install_local_monkeypatches()` — this run is the first confirmation it actually works: the
   fake model got called (`usage.calls=4`) instead of the old silent-no-op failure mode.
2. `ocr_reasoning_extract`'s `words` override param is never exposed by `extract_page` itself —
   `extract.py` calls it via a MODULE import (`from . import ocr_reasoning as OR`), so
   monkeypatching `ocr_reasoning.ocr_reasoning_extract` itself (attribute lookup at call time)
   is a valid injection point for the Molmo wrapper — implemented as
   `_install_ocr_reasoning_molmo_wrapper`, idempotent (installs the untouched original when no
   Molmo points are given).
3. Confirmed real signature: `words` expects raw `(text, x0, y0, x1, y1)` tuples (`idx_words` is
   built internally). The wrapper fetches real OCR words via `ocr_reasoning._ocr_words` (honors
   small-raster auto-upscale + the PaddleOCR patch), builds synthetic rows via
   `molmo_synthetic_tokens`, converts them back into small (±4px) synthetic `Word`-shaped boxes,
   and passes `real_words + synth_words` as the override — going through the exact same
   internal id-assignment/normalization loop as real tokens.
4. `PNID_MODE` handling hardened: `run_one_sheet` now explicitly sets (`"cv"`) or explicitly
   `pop()`s (`"ocr_reasoning"`) the env var rather than relying on it never being set, since one
   process now exercises both configs back to back.

**Acceptance test: PASSED, independently verified (not just trusted from the report) —
read both real output files directly:**
- `L-cv` (sanity re-check, unchanged from Phase 1): `preds_local/GD-T-435-DT-2042-056_p1.json`,
  22 tags, matches prior run.
- `L-ocr` (new): `preds_local_ocr/GD-T-435-DT-2042-056_p1.json` — `drawing.route: "agentic"`
  (confirmed the documented naming quirk, hardcoded for agentic/agentic_tools/ocr_reasoning
  modes), `PNID_MODE` confirmed absent from `os.environ` after the run (real default taken,
  no override), 1 tag with `text: "TEST-201"` (the fake model's constant — correct plumbing
  behavior), `call_llm.usage.calls=4`. revR=0.0 (expected — fake tag doesn't match real truth,
  this is a plumbing test not a quality test).

**Phase 1 + 1b both fully complete.** Both `L-cv` and `L-ocr` local configs are proven to run
end-to-end with fake models.

**Phase 1c — 3 pre-GPU fixes, decided 2026-07-17, before Phase 2.** User asked to give Molmo2
"a real shot" — reviewed 3 flags found on Phase 1b's actual code, verified each against real
pipeline code before approving:

1. **Coordinate-space fix (was: DPI mismatch risk).** Molmo2's pre-pass render must use the
   SAME `triage_page(pg, 0, cfg)` → `RZ.work_zoom(tri.width_pt, cfg)` → `RZ.render_page(pg, zoom)`
   call sequence `extract_page` itself uses internally (verified real, importable:
   `pnid_pipeline/triage.py`'s `triage_page`, `pnid_pipeline/rasterize.py`'s `work_zoom`/
   `render_page` — `work_zoom` is `clamp(11000/width_pt, 3.0, 5.0)`). MUST call `triage_page`
   first, not just `work_zoom` — `tri.width_pt` swaps width/height for rotated pages (confirmed
   real: the acceptance-test sheet is rotated 270°), so skipping triage computes zoom from the
   wrong edge. This eliminates the mismatch by construction rather than correcting for it after
   the fact.
2. **Dedup fix.** Reuse `molmo_candidates.py`'s existing, already-tested `_dedup_points(
   points_by_class, min_dist=radius/2.0)` in the Molmo caching step before calling
   `molmo_synthetic_tokens` — do not reimplement, verified real and reusable as-is.
3. **Latency fix, WITH a companion safety fix.** Lower `cfg["ocr_reasoning"]["dense_chunk_tokens"]`
   in the harness's loaded config dict (default 1500, target ~800 — a plain dict, no agent-source
   edit) and cap `max_new_tokens` in the real (Phase 2) Qwen generate wrapper. Companion fix
   (found during verification, not in the original 3 flags): `build_qwen_call_llm`'s failure path
   currently returns an EMPTY default instance on ANY JSON parse failure — unlike the real
   pipeline's own client, which salvage-parses truncated output to recover whatever tags DID
   complete before the truncation. Capping token length without a salvage step would silently
   zero out an entire chunk's tags on any truncation. Fix: add a salvage step to
   `qwen_call_llm.py`'s exception path — regex out complete `{...}` objects from partial/broken
   JSON before falling back to the empty default. Unit-test with a deliberately truncated fake
   answer (must recover the complete objects, drop only the incomplete tail).

All 3 (+ companion) are CPU/fake-model testable — no GPU or PDF upload needed. Dispatched for
build now.

**Phase 1c complete and independently verified (2026-07-17).** Coordinate fix
(`molmo_render.py`, real `triage_page`→`work_zoom`→`render_page` sequence, self-test
cross-checks against an independent manual replication, confirms the 270°-rotated test
sheet's width/height swap), dedup fix (wired `molmo_candidates._dedup_points` into the
`L-ocr` wrapper, new test proves near-dupes collapse), latency+salvage fix
(`dense_chunk_tokens`→800, `max_new_tokens_cap` threaded through, string-aware balanced-
brace salvage parser in `qwen_call_llm.py` recovers complete tags from truncated JSON).
19/19 tests pass (15 pytest + 4 standalone), independently rerun, not just trusted from
the build report.

**Phase 2 (steps 1-2 of 4) complete and independently verified (2026-07-17).** Built:
- `src/extraction_local/qwen_generate.py` — real Qwen3-VL-8B `generate_fn`, recipe copied
  verbatim from `ArmL_QwenVL_FullStack_GPUOnly.ipynb` (confirmed matches
  `qwen_call_llm.py`'s exact 2-or-3-arg calling convention).
- `src/extraction_local/molmo_points.py` — real Molmo2-O-7B per-class pointing wrapper.
  Config (tile=512/overlap=102/upscale=2/enhance=True) verified against the actual
  logged result in `Stage4_Detection_GPT55_vs_Molmo2.ipynb` (real output:
  `molmo2-points tile=512 up=2 enh=True P=0.6309 R=0.6244 F1=0.6276` — the "F1=0.628"
  citation is a real executed run, not an estimate). Coordinate-space contract
  (must receive `molmo_render_page`'s own output, never a differently-scaled render)
  is explicit and correctly reasoned through in the module docstring.
- `scripts/package_extraction_agent_src_for_colab.sh` (new, separate from the
  intelligence-agent's existing script — confirmed untouched) — has a PDF tripwire,
  written but not executed.
- `notebooks/e2e_harness/ExtractionAgent_Local_GPUOnly.ipynb` (25 cells) — two-phase
  model schedule, smoke gate before full spend, checkpoint section up top. Confirmed
  honest: `L-cv+M` (the not-yet-wired Molmo→CV-hybrid merge) is left as a clearly
  labeled TODO stub, not faked as working.

**Cannot be verified without a GPU** (neither I nor the build agent has one): whether
Qwen3-VL reliably emits valid JSON under these real prompts, whether Molmo2 transfers to
these denser industrial sheets. Both remain open per §9's risk list until an actual run.

**Checkpoint §10 item 1 — RESOLVED, uploads done (2026-07-17), explicit user go-ahead given.**

1. Ran `scripts/package_extraction_agent_src_for_colab.sh` → pushed to
   `timthy45/pnid-extraction-agent-src` (confirmed private=True via API). Verified zip
   contents directly (not just trusted the script's own "done" message): 285 files,
   106 `review_reads/` ground-truth files present, `extraction_local/` present,
   `score.py` present.
2. Zipped `AG_PNID` (63.0MB) and `RIVE_LTTS_Sample` (20.3MB) trees (macOS junk/`.DS_Store`/
   `__MACOSX` excluded) and pushed to `timthy45/pnid-extraction-datasets` at
   `sheets/AG_PNID.zip` / `sheets/RIVE_LTTS_Sample.zip` — the exact paths
   `ExtractionAgent_Local_GPUOnly.ipynb`'s §4 placeholder cell expects. Confirmed private=True
   and both files present via `list_repo_files`, not just trusted the upload's exit message.

**Next: run the notebook.** Fill in §4's placeholder cell (uncomment, point at the now-real
`sheets/*.zip`), open in Colab with a GPU runtime, Run All through the §7.1 smoke gate
only (`SMOKE_ONLY=True` default) — per the notebook's own run-order note, show Tom the
smoke-gate numbers before setting `SMOKE_ONLY=False` for the full 13-sheet run.

**§4 placeholder filled in (2026-07-17) — real bug caught and fixed before it could bite.**
While filling in the download cell, checked the ACTUAL internal path structure of the two
uploaded zips (`unzip -l`, not assumed) and found a real mismatch: the zips are single-level
(`AG_PNID.zip` → `AG_PNID/<file>.pdf`, `RIVE_LTTS_Sample.zip` → `RIVE/<file>.pdf`), but the
notebook's `AG_DIR`/`RIVE_DIR` (copied from `score_revR_real_sheets.py`'s local-scratchpad-
matching paths) expected the double-nested `AG_PNID/AG_PNID/...` shape the LOCAL scratchpad
folders have — not what got zipped. Fixed by correcting `AG_DIR`/`RIVE_DIR` in cell-3 to
`/content/sheets/AG_PNID` and `/content/sheets/RIVE` (matching the real zip contents) rather
than re-zipping/re-uploading. Updated cells 1, 8, 24's markdown to match. Notebook re-validated
as parseable JSON, 25 cells, after all edits.

**Status: notebook fully ready to run.** Nothing left for Sonnet to build — next action is
Tom opening it in Colab.

**`L-cv+M` wiring complete (2026-07-17) — Phase 1 build-log gap item 3 closed, zero
agent-source edits.** The "no existing monkeypatch point" finding turned out to be wrong on
re-read: `extract.py` line 25 does `from .reconcile import assertions, snap_candidates`, and
every `snap_candidates` call inside `extract.py` is a module-global lookup at CALL time — so
monkeypatching `pnid_pipeline.extract.snap_candidates` IS a clean extension point. Built
`_install_cv_molmo_snap_wrapper` in `run_extraction_local.py` (same idempotent
install/restore pattern as `_install_ocr_reasoning_molmo_wrapper`; `_ORIG_SNAP_CANDIDATES`
captured at import):

1. **One-shot injection.** `snap_candidates` runs up to 3×/run (extract.py line 100 first
   merge inside `_path_b_candidates`, line 110 adjudication re-snap, line 246 A+B combined
   re-snap). Verified: routes "B"/"A+B" enter `_path_b_candidates` (line 237–239) before
   line 246, and line 246 is guarded by `if symbols and R` (symbols only exist if Path B
   ran) — so line 100 is always the first call. A closure flag injects
   `molmo_candidates(points, ocr_words, radius=120)`'s candidate dicts exactly once, at the
   real shape+region+ocr merge.
2. **In-place `symbols` extension.** The wrapper `symbols.extend(...)`s (never rebinds) the
   list object `_path_b_candidates` owns, so the same object flows into
   `assertions(symbols, cands, R)` (line 104) and the adjudication loop — the §3.B design
   ("snap can corroborate, assertions counts them") achieved without source edits.
3. **OCR-word capture.** `_paddle_tiled_ocr_words_shim` now stashes its latest return in a
   module holder (`_LAST_OCR_WORDS`, reset per run); OCR runs at extract.py line 78, before
   the first snap call, so the wrapper always sees the run's own words.
4. **Coordinate space verified:** `_path_b_candidates` renders via
   `RZ.work_zoom(tri.width_pt, cfg)` → `RZ.render_page(pg, zoom)` (extract.py lines 74–75) —
   the exact sequence `molmo_render_page` replicates, so Phase A's cached points are already
   in this path's pixel space.

Tests: 3 new pytests (`tests/test_cv_molmo_wiring.py` — first-call-only injection, in-place
`symbols` identity, restore-on-None); full suite 18/18 passing. Fake-backed end-to-end
acceptance on `GD-T-435-DT-2042-056` (`pipeline_mode="cv"`, fake point 10px from the real
PaddleOCR word "TPA-77501" at the CV render zoom=5.0, discovered by actually running
`molmo_render_page`+`paddle_ocr_words`): final preds contain tag `TPA-77501` with signals
`['ocr_word', 'cv_shape', 'molmo_point']` — the Molmo candidate survived validate/assemble
AND picked up `cv_shape` corroboration from its own injected synthetic symbol box.
`n_tags` 22→23, revR unchanged (0.636, plumbing not quality). Notebook cells 0/14/19 updated:
`L-cv+M` now runs `run_config("cv", use_molmo=True, config_name="L-cv+M")` as a REAL
ablation (still never GPU-tested — honesty note kept). One deviation from the wiring spec:
none of substance; the deprecated "small, explicit agent-source change" fallback in §3.B's
table is now moot.

**FIRST REAL COLAB RUN, 2026-07-17 — real bug hit, fixed.** Tom ran the notebook for real
(A100-SXM4-80GB). §2 install cell ran clean, §3/§4 (private code + sheet PDFs) worked as
built. §5 (`load_molmo_model`) crashed:
`ImportError: cannot import name '_Ink' from 'PIL._typing'`. Full traceback showed the real
cause: `transformers` conditionally imports `torchvision` (`image_utils.py`,
`is_torchvision_available()`), which chains into `torchvision.utils` → `PIL.ImageDraw` →
`PIL.ImageText` → `from ._typing import _Ink` — absent from Colab's installed Pillow. A
torchvision/Pillow version mismatch in the Colab environment, unrelated to Molmo2,
Qwen3-VL, or any code in this project. Neither `qwen_generate.py` nor `molmo_points.py`
uses torchvision anywhere (only `torch`/`transformers`/`PIL` directly), so fixed by adding
`!pip uninstall -y torchvision -q` to the §2 install cell — `transformers` skips the whole
broken import path when torchvision isn't present. Notebook re-validated (25 cells, valid
JSON) after the fix.

**Also flagged:** the config cell Tom showed contained a live `HF_TOKEN` pasted in
plaintext — flagged for rotation, same as prior exposed-token incidents this project.

**Attempt 1 fix was WRONG, real root cause found (2026-07-17).** `!pip uninstall -y
torchvision` "fixed" the first traceback but broke a different thing: Molmo2's own
remote-code modeling file (loaded via `trust_remote_code=True`) has a HARD, direct
dependency on torchvision — `transformers.dynamic_module_utils.check_imports` scans the
actual downloaded modeling file's imports and raises loudly if a required package is
absent. This is NOT the same as transformers' own merely-optional torchvision import
(`image_utils.py`'s `is_torchvision_available()` check) that caused Attempt 1's traceback
— torchvision must stay installed for Molmo2 specifically.

Root cause, confirmed via web search (matches independent reports from `ultralytics` and
`unsloth` hitting the byte-identical traceback): **Pillow 12.0.0 removed/changed `_Ink` in
`PIL._typing`**, a known upstream regression breaking any code (torchvision's
`ImageDraw`/`ImageText` usage, triggered here via transformers' image utils) that
references it. A bare `--force-reinstall` doesn't help — it just reinstalls whatever
"latest" resolves to, which is the same broken 12.0.0. Fixed with an explicit version pin:
`pip install -q "pillow<12"`, torchvision left installed. Notebook re-validated (25 cells,
valid JSON).

**3rd iteration (final, thorough) — 2026-07-17.** Attempt 2's rerun failed with
`ModuleNotFoundError: No module named 'torchvision'` because of a Colab gotcha the 2nd fix
missed: **`Runtime → Restart session` only restarts the Python kernel — pip changes live on
the VM's disk and survive restarts.** Attempt 1's `pip uninstall torchvision` therefore
persisted across the restart; nothing was going to bring it back implicitly. Final §2
install cell now does all three things in the right order, plus verifies itself:
1. Reinstalls torchvision WITH torch pinned to its currently-installed version
   (`pip install torchvision "torch==<current>"`) so pip resolves the paired torchvision
   build instead of upgrading torch itself (torchvision releases hard-pin exact torch
   versions; an unconstrained install would drag in a new torch and risk the CUDA stack).
2. Pins `pillow<12` LAST (after torchvision/paddle installs, which can each pull Pillow
   back up to the broken 12.x).
3. **Fail-fast verification block**: re-executes the exact import chains that crashed §5
   in the real runs (`from PIL import ImageDraw`, `import torchvision`,
   `from transformers import AutoModelForImageTextToText, AutoProcessor`,
   `import paddleocr`), asserts Pillow is not 12.x, and raises with a clear restart
   instruction if torch's on-disk version diverged from the in-memory one — so any
   remaining environment problem surfaces in §2 with a precise message, never again 3
   cells later inside model loading.

Notebook re-validated (25 cells, valid JSON). **Status:** thorough fix applied, awaiting
Tom's rerun (Runtime → Restart → Run All through §7.1).

**FIRST REAL RESULTS (2026-07-17, deadline run — 3 dev sheets, L-ocr config only).**
Full detail in `results.csv` row `extraction_agent_L-ocr_qwen3vl_3sheets`. Headline:
GD-T-435-DR-2031-030 **0.810 vs GPT-5.5-high 0.841 (96%, near-parity)**;
PX-2365-0150022-001 0.183 vs 0.75; PX-2368-0180004-001 0.140 vs 0.98. Three
diagnosis-driven prompt rounds (schema-echo → id-dump → category coverage), each fixing
a real observed failure; PX gap isolated to compound-tag/suffix read precision (a
fine-tuning target), not extraction strategy. OCR served from Mac-precomputed cache
(Colab paddle segfaults natively at every version tried). Molmo2 configs deferred
(Phase A runtime infeasible under deadline) — all wiring built and fake-tested, ready
for the follow-up run. Environment gauntlet (7 real Colab failures, all root-caused):
documented across this section. Remaining from the original protocol: full-13-sheet run,
L-cv arm, Molmo ablations, results write-up per §8.
