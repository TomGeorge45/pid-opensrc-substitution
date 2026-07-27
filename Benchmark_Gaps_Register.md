# 3-Way Relation-Stage Benchmark — Gaps Register

*2026-07-23. The complete list of gaps standing between us and a full-fledged, honest
capability benchmark of the three pipelines (pnid-extraction-agent with proprietary
models · same architecture with local models · proposed R1–R5 pipeline), split by who
can close each gap. Source: verified code reads of both agent repos + PR #711 + this
week's benchmark sessions. Numbers refer to the original 21-gap analysis.*

---

## Group 1 — I close these (pure engineering/protocol; no input needed)

| # | Gap | What I'll do |
|---|-----|--------------|
| 4 | PID2Graph GT routes edges through connector/crossing nodes — unusable raw | Build the deterministic graph-contraction step (collapse connector/crossing/arrow chains → symbol↔symbol edges), freeze the rules, document them |
| 6 | Tag-less GT nodes are ungroundable for the LLM arms | Render annotated overviews with entity IDs drawn on boxes (stage-13's own format) + coordinates in the tag list; documented as a deviation from prod |
| 7 | GT classes must map to each pipeline's type vocabulary | **Done (2026-07-23):** `src/relation_bench/type_vocab.py` — frozen `valve→valve, instrumentation→instrument, tank→equipment, pump→equipment, general→equipment` against extraction-agent's real `Tag.type` taxonomy (confirmed via direct code read of `pnid_pipeline/agentic.py::ASSET_TYPES`). Wired into the gap-#12 shim's `_tags_from_gt_nodes` — previously fed PID2Graph's raw class name straight into `Tag.type` unmapped, which would have silently biased pipeline 1/2: e.g. `hierarchy.py`'s system-member relation only fires on the literal string `"equipment"`, so an un-mapped `"general"` would have made that pass never fire for reasons unrelated to the LLM's reasoning. Self-tested: 93 injected tags now bucket correctly as `{equipment: 22, instrument: 36, valve: 35}`. Pipeline 3 (R2a/R2b) deliberately needs no mapping — designed from the start to consume PID2Graph's own class names directly. **Flagged for your review per the original ask — a bad mapping could still bias the pipeline-1/2 arms; the tank/pump/general→equipment collapse in particular is a judgment call worth a second look before real arms run** |
| 8 | No edge-scoring harness exists | Build it (undirected pair matching, dedup, per-stratum reporting) |
| 9 | R1 raster line-tracer not ported | Port from intelligence-agent Stage 6 |
| 10 | R2a deterministic builder not ported | **Done (2026-07-23):** ported Stage 11's 3-pass `build_relations` (inline-chain, direct endpoints, hop-capped BFS traversal) as topology-only `build_topology_relations` — ontology relation-naming stripped since this benchmark scores existence, not kind (`src/relation_bench/graph_construction/`). Also ported the junction-to-symbol-boundary map. Self-test on OPEN100/0: R1+R2a F1=0.104 vs R1-alone F1=0.023 — traversal recovers real edges as designed (tp 4→20), still below the nearest-neighbor floor because R1's PID2Graph ceiling (gap #9-confirmed) caps everything downstream. Hop depth 8 vs 20 vs 40 made no difference on this sheet. **Not ported:** the ISA valve-operator authority rule and loop grouping (Stage 10.5) — the ISA rule parses tag TEXT, which PID2Graph nodes don't have (class-agnostic, like Gupta), so it's genuinely untestable here; only worth vendoring once the AG/RIVE fixture (Group 2) exists. Loop grouping (Stage 10.5, equipment-skid grouping via Opus 4.7) is a separate, heavier stage judged out of scope for a relation-topology benchmark — flagged, not silently dropped |
| 11 | R2b hierarchy geometry pass designed but unwritten | **Done (2026-07-23):** built `src/relation_bench/hierarchy/` — containment prior (child bbox nested in an equipment bbox) + connectivity-nearest prior (R2a-adjacent equipment, nearest by bbox-center distance), then a verbatim port of prod's `_break_cycles` (tie-break swapped from shortest-tag-text to smallest-bbox-area since PID2Graph nodes have no text). Self-test on OPEN100/0: 0 cycles remain (valid forest), 13/93 nodes parented, all via connectivity-nearest (no containment matches — P&ID symbols rarely nest bboxes on this corpus), child→parent class pairs all sane (valve→general, instrumentation→general). Low parented-count is the same R1-ceiling cascade already documented, not a new bug. **Cannot be scored with P/R/F1 yet** — no hierarchy ground truth exists on PID2Graph (Group 2 gap #2, still unanswered); this is a structural self-test only |
| 12 | No standalone shim for pipeline 1/2's real relation code | **Done (2026-07-23):** `src/relation_bench/arms/hierarchy_shim.py` runs the REAL, unmodified `pnid_pipeline.hierarchy.apply_hierarchy` on injected GT entities, exact same proven pattern as `run_one_sheet` (real prod code, swappable `call_llm`, zero agent-source edits). Both real `call_llm` builders already exist and match the exact contract `apply_hierarchy` expects — no new client code needed: arm 1 (prod) = `pnid_pipeline.llm_proxy.build_call_llm(model="gpt-5.5-low")` (confirmed via direct code read: the proxy already routes a gpt-5.5 model name through the same `/v1/messages` shape), arm 2 (local) = `extraction_local.qwen_call_llm.build_qwen_call_llm(generate_fn)`. Also built `topology_pairs_from_result` so pipeline 1/2's hierarchy output scores through the exact same `score.py` harness as pipeline 3. **Self-tested wiring only** (fake call_llm, empty default-instance replies — zero real tokens spent): both real LLM call sites fire correctly (hierarchy pass + chunked connectivity pass), 93/93 tags pass through, no crashes, all-zero score as expected from an empty reply. **Honest limitation surfaced by building this:** PID2Graph GT nodes carry no tag text, but prod's hierarchy pass is built entirely around tag-text reasoning (`_nesting_parent` prefix matching, ISA loop hints, the LLM prompt's `id\|TEXT\|type\|cx,cy` lines) — feeding it text-less entities strips out the nesting prior entirely and leaves the LLM almost nothing but a class label + position. This is a real, reportable degraded-input condition every arm faces on PID2Graph, not a shim bug — and it's exactly why the AG/RIVE annotated fixture (Group 2, gap #3+5) matters: only there do arms 1/2 get real tag text to reason over. **Still pending, and NOT something to do unprompted:** actually running real GPT-5.5-low / local-Qwen calls through this shim costs real API/GPU time — that execution decision belongs to Tom |
| 13 | v3-relation's training sheets never saved as a list | **Done (2026-07-23), resolution: v3-relation dropped from this benchmark.** Replayed the real notebook's exact seeded sampling (`random.Random(777)` for training, `random.Random(4242)` for the mid-training eval probe — same call order/state as `Stage4_PerStage_Stage13_and_Relation_v3.ipynb` cells 20+26, not just a file diff) against the actual local PID2Graph tree. Also resolved the `Patched/` vs `Complete/` tree question directly: `Patched/` holds ~140 augmented/jittered variants per source sheet (training volume), `Complete/` holds the one canonical graphml+png per sheet (what R1/R2a/R2b were already correctly self-tested against) — training draws from `Patched/`, scoring should always draw from `Complete/`. **Finding: both trees are fully exhausted.** OPEN100 has only 1,629 total patches across 12 source sheets against a training cap of 1,600 (98% consumed) → all 12/12 sheets touched. Dataset PID has ~39 patches/sheet on average against a 4,000-patch cap → all 500/500 sheets touched. Confirmed via `HfApi`/direct zip inspection this isn't a partial download — the team's HF mirror (`timthy45/pnid-extraction-datasets`) genuinely contains only these 12+500 sheets, nothing more to pull down. Checked whether the eval-probe sheets (seed 4242) could serve as a substitute held-out set — rejected: they're a subset of the SAME sheets training touched (6/12 OPEN100, 99/500 Dataset PID), and OPEN100's ~140-patches-per-sheet training volume means the model saw extensive overlapping crops of every one of those sheets, so a held-out *pair* on an already-trained-on *sheet* isn't a clean generalization test — same sheet-level discipline as CLAUDE.md rule 7's frozen Gupta test sheets. **Tom's decision:** drop v3-relation from this benchmark entirely. Pipeline 2's ("local arm," gap #12) relation stage uses zero-shot/domain-base Qwen instead of the v3-relation checkpoint — no fine-tuning on any PID2Graph sheet means no contamination, and every currently-local sheet becomes usable as held-out immediately. Revisit v3-relation scoring later only if a larger, independently-sourced PID2Graph release turns up |
| 14 | Process-backbone pass unbuilt (anywhere) | Buildable by me (deterministic graph walk through inline pass-through classes) — scoped as post-benchmark upgrade #1; for the benchmark itself, pipeline 3's asset↔asset ceiling gets reported as known-limited |
| 16 | Direction convention undecided | Freeze: score undirected pairs (GT direction isn't reliable process flow); documented in the harness |
| 17 | Sample size/stratification undecided | Freeze: 12–15 held-out sheets, stratified sparse (OPEN100) vs dense (Dataset PID). No exclusion list needed anymore — v3-relation dropped (gap #13), so no arm in this benchmark was fine-tuned on any local PID2Graph sheet; every sheet is fair game |
| 18 | LLM-arm run-to-run variance unbounded | Harness supports N repeat runs on a subset; **execution on Colab GPU is your usual role** |
| 19 | Flat means hide failure modes | **Done (2026-07-23):** per-stratum reporting live in `score.py` on all three axes — line type and endpoint-class pair were already built into `score_topology`; sheet-density added today as an optional `sheet_group` label per sheet ("sparse" = OPEN100, "dense" = Dataset PID, the gap-#17 split). Density is a sheet-level property so the whole sheet's tp/fp/fn goes into one `sheet_group:<label>` bucket; `aggregate()` needed no changes (it already sums stratum keys generically across sheets). Self-tested on a real 2-sheet aggregate (OPEN100/0 sparse + Dataset PID/0 dense), and the stratification immediately paid for itself with two findings a flat mean would have hidden: (1) R1+R2a scores **better on the dense sheet** (F1 0.202 vs 0.104 sparse) — dense sheets' shorter pipe runs are likelier to survive R1's fragmented tracing, so the sparse tree is the harder case, not the easier one; (2) `valve\|valve` is by far the strongest class stratum (F1 0.314 vs 0.126 asset↔asset) — adjacent inline valves connect through short segments, while asset↔asset needs the long multi-hop backbone runs R1 loses, consistent with the process-backbone gap (#14) being pipeline 3's known weak axis |
| 20 | No cost/latency accounting | **Done (2026-07-23):** `src/relation_bench/cost_latency.py` — one `CostLatencyRecord` per (arm, sheet), appended uniformly to `relation_bench_results.csv`. Reuses real prod utilities rather than inventing new tracking: `pnid_pipeline.llm_proxy.snapshot`/`delta`/`usage_cost` work unmodified on EITHER real `call_llm` (prod's `build_call_llm` and local's `build_qwen_call_llm` maintain the identical `.usage` dict shape, confirmed by direct code read) — `usage_cost` (the real $ rate table) is applied only for the prod arm; the local arm's cost is always $0 regardless of token usage, since local GPU inference isn't billed per token. Pipeline 3 (R1/R2a/R2b) has no LLM calls at all, so its cost is always $0 — only wall-clock latency matters there, same convention as CLAUDE.md's existing `results.csv` schema (`vram_gb,latency_s_per_tile`, no $ column for local compute). Self-tested end-to-end on OPEN100/0: pipeline 3's full R1+R2a+R2b run timed at 12.0s/$0; the arm shim's fake call_llm timed at 0.036s/2 calls, correctly $0 as the local arm and correctly converted to a real dollar figure ($0.009) on the identical usage when scored as the prod arm — confirms the cost-conversion path itself works, not just the timer |
| 21 | No sanity floor | Trivial nearest-neighbor-connect baseline included |
| 22 (new) | Off-page connector references make most real LLM connectivity claims structurally unverifiable by any single-sheet tracer | Surfaced by the Fable adjudication (2026-07-23) on PX-2368-0180004-001: of ~25 true GPT-5.5-low claims, ~19-20 had their remote endpoint drawn only as a text label inside a "FROM/TO ... (sheet-number)" border annotation, not as a real symbol on this sheet at all — no tracer, raster or vector, can ever confirm those, since the referenced equipment isn't drawn here. This is why the agreement-diff read as near-zero even though GPT-5.5-low was ~80% accurate. **What I can close (Group 1):** partition every LLM claim into on-sheet↔on-sheet vs on-sheet↔off-page BEFORE diffing against traced geometry — only the former is fairly tracer-comparable; the off-page half needs a different check entirely (does the claim correctly name the off-page equipment the connector arrow points to, checkable from the annotation text alone, no tracing required). **What's a bigger, separate undertaking, not close-able as a quick fix:** true cross-sheet verification — actually following the referenced sheet number to confirm the line reaches a matching connector there — needs the full linked drawing set for a job (not just isolated single sheets) and a cross-sheet linker that doesn't exist yet anywhere in this project. Scoped similarly to gap #14 (process-backbone pass): a real, valuable post-benchmark upgrade, not blocking this round |

## Group 2 — needs an answer or hours from you

| # | Gap | What I need from you |
|---|-----|----------------------|
| 15 | Which "prod" is arm 1? | **Decide:** GPT-5.5-low (verified prod, OpenAI key works locally) or Sonnet 4.6 (repo config default) — or run both. If Sonnet: **is an Anthropic API key available?** |
| 3+5 | No joint tag+position+edge GT on real sheets; PID2Graph has no PDFs so the vector fast-path (pipeline 3's biggest lever) is untestable there | **Decide + ~2–4 h of your review time:** manually annotate 2–3 AG/RIVE sheets (boxes + tags + edges) using annotation aids I'll prepare (the overlay tooling doubles for this). I should NOT author the ground truth myself — I'm a party being benchmarked; GT must be human-verified |
| 2 | No hierarchy/containment GT anywhere — stage-6-equivalent output unscoreable for ALL three pipelines | **Decide:** add parent/child labels to the same 2–3 annotated sheets (small extra effort on top of 3+5), or accept hierarchy as unmeasured in round 1 |
| 1 (partial) | No relation-KIND ground truth (`feeds` vs `actuates` vs `signal_to`) | **Decide:** add kind labels to the annotated fixture edges (needs P&ID-literate judgment — yours or an engineer's), or accept topology-only scope for round 1 |
| — | Confirm the frozen type-vocabulary mapping (Group 1, #7) before arms run | Quick review of one table I'll produce |
| 23 (new) | Full 492-candidate R4 (local Qwen relation-validator) run — recommended to skip, not confirmed | **Decide:** the 8-candidate/sheet smoke gate (2026-07-25) already showed R4 rejects 100% of candidates on all 3 real sheets, both configs (Part D). My recommendation is skip the full run — it would almost certainly just reconfirm the same zero result, not worth the GPU time — but this was my call, not an explicit "yes skip it" from you. Say go/no-go either way |
| 24 (new) | `PASSTHROUGH_TAG_TYPES = {valve, instrument, fitting, safety_device}` in `relationship_pipeline.py` — frozen, unreviewed | **Decide:** same pattern as the Group 1 #7 type-vocab mapping — a quick review of whether this is the right set of real-drawing `Tag.type` values to walk through for the backbone pass (Part D) |
| 25 (new) | Whether to wire the backbone pass into `agreement_diff.py`'s AG/RIVE path | Currently the backbone pass only runs in the PID2Graph path (Part D); extending it to AG/RIVE changes what the Part B agreement numbers mean. **Ask before wiring it in** |
| 26 (new) | Whether the entity-extraction stage's Molmo2+`tag_matching.py` nearest-OCR-word design (real symbol-shape location, not just tag-text location) should be wired into relationship-stage testing | Raised mid-session 2026-07-27, not decided. Would directly address the recurring tag-text-vs-symbol-shape bug (Part D bottom line) at its source, rather than working around it downstream. Cross-cutting with the entity-extraction multi-arm architecture (`E2E_Harness_Plan.md`) — needs your call on priority, not something to build unprompted |

## Group 3 — genuinely uncoverable right now

| # | Gap | Why it can't be closed |
|---|-----|------------------------|
| 5 (PID2Graph side) | Vector fast-path can never be tested on PID2Graph itself | The dataset ships PNGs only; the source PDFs don't exist anywhere we can get them. The capability is only testable via the AG/RIVE fixture (Group 2) — on PID2Graph it stays untested, period. **Confirmed via a dedicated dataset search (2026-07-23), not just assumed:** checked the actual PID2Graph Zenodo release + its source paper (Stürmer et al. 2025, arXiv:2411.13929) + Digitize-PID (Paliwal et al. 2021, arXiv:2109.03794) directly. Our local 512-sheet mirror (12 OPEN100 + 500 Dataset-P&ID) *is* the complete public release — OPEN100's "100" is not a sheet count (the real-world subset is genuinely 12 sheets everywhere it's published), and Dataset-P&ID's 500 synthetic sheets is Digitize-PID's full release. No larger/richer PID2Graph exists to go fetch. Also checked for any other public P&ID dataset with real tag text + relation-kind + hierarchy + source PDFs + >512 sheets — none found (the one near-miss, Enginuity, is 50K+ automotive/mechanical engineering diagrams, not P&IDs — wrong domain, not usable). The AG/RIVE hand-annotation path (Group 2) is confirmed as the only route to closing this, not one option among several |
| 1 (at scale) | A statistically powered relation-KIND benchmark | A few dozen hand-labeled edges (Group 2) gives a signal, not statistical power. Hundreds of kind-labeled edges across many sheets would need annotation resources that don't exist right now |
| — | Prod's actual relation-stage quality baseline | Production has never scored or logged its relation outputs (scoring functions exist in `score.py`, never called; no GT in any fixture). There is no historical number to compare any arm against — the benchmark creates the first-ever measurement, but "did prod get worse/better over time" is unanswerable |
| — | Line-typing (process vs signal/utility) from vector strokes on this corpus | PR #711 proved it dead: the CAD exports flatten everything to solid single-color overlapping-width strokes — 0% dashed. Only raster stroke-width sampling or topology heuristics could substitute (future work, unproven) |
| 9 (confirmed) | R1's raster line-tracer has a low recall ceiling on PID2Graph specifically | Ported verbatim (`src/relation_bench/line_tracing/`), self-tested on OPEN100/0 (93 symbols, 334 contracted GT edges): best config found only 13 predicted symbol↔symbol pairs, F1=0.023 — *below* the nearest-neighbor floor (F1=0.226). A 3-point threshold sweep (snap radius, bridge gap, mask pad) made it worse, not better, ruling out simple tuning. Root cause: PID2Graph's node ontology has no text/tag class, so tag labels and line annotations near pipes stay unmasked and fragment the skeleton into spurious junctions/loose-ends — real Stage 6 avoids this because Stage 4 detections mask text-tag boxes too, which PID2Graph doesn't provide. Consistent with PR #711's own finding (raster-only path recovered ~4 asset relations before the vector fast-path existed). **Tested the text-masking hypothesis directly (2026-07-23):** ran PaddleOCR on the same sheet, masked all 202 detected text boxes as extra_mask_bboxes alongside the symbol boxes — result got *worse* (F1 0.012 vs 0.023), and segment/junction counts collapsed (900→170, 267→78), meaning the masking removed real line content, not just tag noise. This rules out missing-tag-boxes as the cause. Root cause is a deeper corpus-rendering mismatch between PID2Graph's raster and what Stage 6's CV was tuned against — not something more masking or threshold tuning fixes. Report R1 honestly as a known-low-ceiling raster baseline on PID2Graph specifically; not a bug to keep chasing |

---

## Part B — real-sheet agreement audit (AG/RIVE PDFs) — in progress, 2026-07-23

Re-downloaded AG_PNID (10 sheets) + RIVE_LTTS_Sample (20 sheets) fresh from the HF mirror
(prior session's copies had been cleaned up as ephemeral scratch). Confirmed via direct
PyMuPDF inspection: genuinely born-digital vector PDFs (thousands of real drawing paths per
page), not scanned rasters — the first time this project can exercise the vector fast-path
at all (PID2Graph structurally can't, confirmed via the dataset search above).

**Vector-geometry tracer — built and self-tested, free/local, no LLM cost:**
- Vendored `vector_graph.py` (PR #711's real fast-path builder) into
  `src/relation_bench/line_tracing/`, plus two pure bbox-geometry helper functions
  extracted standalone from `isa_rules.py` (`diagram_area_bbox`, `bbox_center_outside`) —
  avoided the ~650-line tag-grammar dependency chain entirely since neither function
  touches it.
- Built `pdf_vector_extract.py`: turns PyMuPDF's real drawing commands (lines, bezier
  chords, rect/quad edges) into the flat segment list the tracer expects.
- Self-tested on `GD-B-540-DP-2920-005-Z.pdf`: 39,135 raw vector segments → 3,477 topology
  segments after bridging/noding/contraction; with OCR words standing in for symbols
  (wiring test only), 449 symbol nodes resolved, 996 segment endpoints correctly snapped.

**Real entity extraction — built and run for real, 3 sheets, real GPT-5.5-low cost:**
- Built `arms/openai_call_llm.py`: a new, direct GPT-5.5-low `call_llm` adapter matching
  `apply_hierarchy`'s exact contract. Needed because the originally-assumed adapter
  (`pnid_pipeline.llm_proxy.build_call_llm`) talks to an Anthropic-Messages-shaped proxy
  that isn't running here (only a raw `OPENAI_API_KEY` is available) — confirmed live: a
  raw `model="gpt-5.5-low"` string 400s ("model does not exist"); the real split is base
  model `"gpt-5.5"` + a separate `reasoning={"effort": "low"}` field, matching how
  `real_openai_client.py::RealOpenAIRunner` already drives GPT-5.5-low elsewhere in this
  repo. Usage tracked in the same shape `llm_proxy.snapshot`/`delta`/`usage_cost` (gap #20)
  already expect, so cost accounting is uniform with no special-casing.
- **Real credential gap found and worked around, not silently patched:** prod's OCR step
  needs `GOOGLE_CLOUD_VISION_API_KEY`, which we don't have — first attempt silently
  returned 0 words → 0 LLM calls → 0 tags, no error surfaced (prod's own try/except
  swallows the OCR failure). Fix: since these are real vector PDFs, PyMuPDF's own embedded
  text layer (`page.get_text("words")`) is genuine, perfect text — actually better than OCR
  would ever be here — and prod's own code already treats vector PDF text as a legitimate
  token source (`PNID_HYBRID_TOKENS`). Substituted it directly for the unavailable OCR
  call (documented as a deviation in `run_real_extraction_partB.py`'s docstring), running
  the REAL `ocr_reasoning_extract` + `apply_hierarchy` + `build_result` functions unmodified.
- **Ran on 3 real sheets** (user-approved small batch): PX-2368-0180004-001 (120 tags, 179
  relationships, $0.4493, 4 calls), GD-B-540-DP-2920-005-Z (111 tags, 161 relationships,
  $0.2955, 3 calls), PX-2365-0140006-001 (436 tags, 618 relationships, $1.1407, 8 calls).
  **Total: 667 tags, 958 relationships, $1.89, 15 LLM calls.** All real tag text, real
  hierarchy parent_id assignments, zero errors.
- **Real finding, not assumed:** vector-text-layer coverage varies significantly by sheet.
  PX-2368 had 942 extractable embedded words; GD-B-540 had only **1** — that CAD export
  outlines its text into vector paths rather than real font-encoded text objects. GPT-5.5
  still read GD-B-540's tags correctly (`GD09-540-T-9450`, `MV-141`, etc.) from the vision
  image alone, no text-token anchoring assist — the model's OWN vision reading is doing
  real work here, not just token-anchored transcription. Echoes PR #711's earlier finding
  that CAD exports flatten other properties too (stroke width/color); text encoding is
  evidently just as inconsistent across sources.

**Agreement-diff — built, self-tested, and it surfaced a real, honest headline finding.**
`src/relation_bench/agreement_diff.py` compares two independent real signals — never scored
one against the other as if either were ground truth (the tracer has its own known failure
modes, per R1's PID2Graph finding above) — reported symmetrically both directions
(`agreement_rate_llm`, `agreement_rate_geometry`).

- **Caught and fixed a real design bug before trusting the first result:** prod's
  `relationships` field mixes hierarchy/containment relation kinds (`hosted_by`, `on_line`,
  `system_member`, `loop_member` — parent-child asset breakdown) with genuine physical-
  connectivity claims (`feeds`, `relieves_to`, `actuates`, `signal_to`, from the dedicated
  connectivity pass). Diffing ALL kinds against traced line geometry gave near-zero
  agreement even though entity resolution overlapped well (78/79 and 132/139 touched-id
  agreement on two sheets) — the entities were right, the relation KIND being compared
  against a line-tracer was wrong. Fixed: only `CONNECTIVITY_KINDS` are diffed now.
- **Even after that fix, agreement stayed genuinely low.** Multi-hop reachability testing
  (walking through intermediate symbols, up to 15 hops — ruling out "it's just a longer
  path than 1 hop") found only 2 of 31 claimed pairs on the first sheet reachable AT ALL.
- **Traced the cause with an overlay, not just a number** (`overlay_render.py`, same
  disagreement-visualization technique that resolved the earlier v1-vs-zero-shot Molmo2
  question this session): rendered PX-2368-0180004-001 and found nearly every disagreement
  fanned out from ONE hub entity, `MBD-0100` (Bulk Oil Surge Vessel) — GPT-5.5-low correctly
  read it as the process hub multiple off-page connector lines feed into/out of (a
  structurally correct P&ID reading), but its tag bbox is a tiny 130×25px text label sitting
  apart from its actual drawn ellipse outline, so no real pipe endpoint ever resolved close
  enough to it.
- **Tried a targeted fix** (inflate equipment-class tag bboxes by 1 inch of real page space
  before symbol resolution, `EQUIPMENT_BBOX_PAD_INCHES`): partially validated — MBD-0100
  gained 1 more correctly-traced connection on sheet 1 — but did NOT meaningfully move the
  needle overall (GD-B-540 stayed at exactly 0 agreement; PX-2365 improved only 0→2). Same
  pattern as R1's PID2Graph tuning sweep: a real, partial, insufficient fix, not chased
  further past this point.
- **The honest bottom line, a genuinely new measurement no one had before tonight:**
  GPT-5.5-low's real connectivity claims and the PDF's own traced vector geometry
  substantially disagree on real production sheets, even when both sides are looking at the
  same real entities. Some of that gap is a known, partially-fixable entity-resolution
  artifact (equipment bbox ≠ symbol shape); the rest is not yet explained, and likely
  intersects with the still-unbuilt process-backbone pass (gap #14) — the LLM's claims read
  like process-level assertions a single-hop-per-symbol tracer isn't built to confirm.

**Fable third-signal adjudication (2026-07-23, same session, model switched to Fable 5):**
every one of PX-2368-0180004-001's 31 GPT-5.5-low connectivity claims judged by eye against
the real drawing (full page + zoomed crops of the MBD-0100 vessel and NBK-0300 treater
regions), explicitly as a third independent opinion — NOT ground truth (no model's reading
is; that principle stands).
- **~24-25 of 31 claims are process-TRUE per the drawing** (~80% precision). Every "X feeds
  MBD-0100" hub claim that started this investigation corresponds to a real drawn off-page
  connector arrow ("FROM GLYCOL CONDENSATE SEPARATOR (MBD-0635)", "FROM LP DEGASSER
  (MBD-4150)", …) whose line physically merges into the vessel's 18"-245 PSIG inlet header.
  The PSV relieves_to claims match the real 10"-AH flare headers exiting "TO LP FLARE
  SCRUBBER (MBF-0500)". GPT-5.5-low was NOT hallucinating — it read the sheet like an
  engineer.
- **~6 claims are genuinely wrong or unsupported:** 2× "XSY-0301 actuates SDV-0300B/C"
  (no drawn signal line — XSY-0301A points at the transformer; invented from ISA
  tag-naming convention), 2× "XFMR-0301 feeds PBA-0201/0202" (identity error — the real
  drawn pair is NBK-0300↔pumps, which the TRACER found and the LLM missed under that name),
  "HBG-0335 feeds XFMR-0301" (direction/identity muddled — the treater outlet feeds the
  cooler), "MBF-0500 feeds HBG-0110" (links two off-sheet entities; not on this sheet).
  2 more claims have reversed direction (pumps→vessel vs the drawn vessel→pump suction) —
  forgiven by the frozen undirected convention but real errors for any directed round.
- **Why agreement was 3/31 despite ~80% claim precision — the category-mismatch math:**
  of the ~25 true claims, ~19-20 have their remote endpoint OFF-SHEET — equipment that
  exists on this sheet only as text inside an off-page connector annotation. Drawn-line
  tracing can structurally never confirm those against an equipment-symbol endpoint. Only
  ~6 true claims have both endpoints physically drawn on-sheet, and there the tracer
  confirmed 3 (the 3 "agree" pairs). The near-zero agreement is a measurement-category
  artifact, not an accuracy problem on either side.
- **geometry_only pairs adjudicated too:** mostly (a) instrument-inside-vessel pairs the
  LLM correctly files under hosted_by (hierarchy) rather than feeds — a classification
  difference, not a miss; (b) equipment↔line-LABEL pairs (e.g. `6"(300#)`) the LLM by
  design never claims as equipment endpoints; (c) at least one genuine LLM miss the tracer
  caught (PBA-0202↔NBK-0300, the pump-discharge-into-treater line).
- **Concrete fixes this implies for the agreement-diff before it's meeting-ready:**
  (1) partition LLM claims into on-sheet↔on-sheet vs on-sheet↔off-page before diffing —
  only the former is tracer-comparable; the off-page channel is separately checkable by
  tracing from the connector ARROW (not the remote tag text) to the claimed on-sheet
  entity; (2) drop line-label tags from the traced side's endpoint set (treat as
  pass-throughs); (3) note that GPT-5.5-low's real error modes found here (invented
  actuations from tag naming, symbol-identity confusion) are exactly what the proposed
  pipeline's deterministic checks (R2a ISA rules, entity grounding) target.

---

## Part C — hybrid-architecture verifier probes (2026-07-24)

The Fable adjudication (Part B, 2026-07-23) produced a candidate fix for gap #22: instead
of asking an LLM to extract connectivity for a whole sheet, let the vector tracer propose
candidate edges and have a VLM verify each one via a short, bounded yes/no question on a
crop — never open-ended whole-sheet extraction. This section is the go/no-go test of that
architecture's load-bearing assumption: is *any* currently-available model actually good at
the bounded verify-one-edge task on REAL AG/RIVE crops (not PID2Graph, where the ~80-84%
reference figures came from)?

**Test harness:** `probe_bundle_2026-07-24.zip` (pushed to
`timthy45/pnid-extraction-datasets`, `benchmarks/`) — 19 valid (20 minus 1 SKIP) hand-verified
real symbol pairs across the 3 AG/RIVE dev sheets (`PX-2368-0180004-001`,
`GD-B-540-DP-2920-005-Z`, `PX-2365-0140006-001`), each rendered as a crop with entity A boxed
RED and entity B boxed BLUE, 9 TRUE / 10 FALSE. Pass bar: 80%, matching the PID2Graph
reference figures being tested against. Companion `probe3_answer_key.json`: 8 real off-page
connector-box crops, exact-text answer key, addressing the off-page half of gap #22.

**Probe 1 (symbol-extent reconstruction from vector geometry): PASS**, validated earlier this
session including the harder branching case.

**Probe 3 (off-page connector text reading): PASS, 87.5% (7/8).** Confirms Qwen (and by
extension any comparable VLM) can close the off-page half of gap #22 by reading the
connector-box text directly — no tracing required for that channel.

**Probe 2 (bounded connectivity verification) — FAIL, base Qwen3-VL-8B: 52.6%.** Five rounds
of testing, run in-kernel on `notebooks/e2e_harness/PartB_Qwen_RelationRun_GPUOnly.ipynb`
(sections 9-14), needed to separate a real capability finding from artifacts:

1. **Forced one-word YES/NO:** constant `NO` on all 19 pairs — 52.6% is coincidental
   (exactly the FALSE-class base rate), not discrimination.
2. **Diagnostic (describe-the-boxes-you-see):** Qwen correctly reads both box colors and
   entity tag text — ruled out a rendering/encoding bug.
3. **CoT prompt, no token budget fix:** flatline broke (6/19 said YES) but 10/19 truncated
   mid-reasoning at `max_new_tokens=200` before ever reaching a verdict — mechanical issue,
   not yet a clean signal.
4. **CoT v2 (`max_new_tokens=350`, forced explicit red↔blue endpoint-match check):**
   truncation mostly fixed (2/19 left); **clean result: 9/17 = 52.9% on the parseable
   subset — genuine chance level.** Failure mode: confident, plausible-sounding hallucinated
   traces that never actually check whether the traced path reaches the specific blue-boxed
   target, not incoherent output.
5. **The fine-tuned `v3-relation` adapter (clean test — never trained on AG/RIVE sheets, so
   the PID2Graph contamination concern from gap #13 doesn't apply here):** CoT prompt →
   degenerate `"No.No.No..."` repetition (format mismatch with its narrow training
   distribution, discarded); **its own exact trained prompt format
   (`PerStageV3_Stage13_Relation_vs_GPT55.ipynb` cell 8's short bracket-coordinate style) →
   still degenerate `"No."` repetition verbatim, 0/19 said YES.** Reconciled against its
   earlier 89.2% figure (`E2E_Harness_Plan.md`): that number is real but was measured on
   PID2Graph crops, the adapter's own training domain — it does not generalize to real
   AG/RIVE production sheets, compounded by being frozen at only 1/3 training epochs
   (paused mid-epoch-0, never resumed).

**GPT-5.5-low ceiling test (same 19 pairs, run locally, `.venv-e2e`, ~38 API calls):**
answers the question "is this a Qwen-specific gap, or is the task itself too hard for any
model at this scale?" — **68.4% (short prompt) / 73.7% (CoT v2), both below the 80% bar.**
GPT's errors are high-precision/low-recall (CoT: said YES only 4 times, correct all 4; every
error was a missed true connection) — the wrong bias for a verifier meant to filter
candidates without discarding real edges.

**Eyeball audit of the 4 pairs every model missed (pair_01, pair_10, pair_14, pair_18) —
surfaced a probe-bundle artifact, not a universal task failure:** all four share one defect —
**the colored boxes highlight tag TEXT, not the drawn symbol**, the same MBD-0100-style
bbox-vs-shape mismatch already documented in Part B's agreement-diff finding. `pair_18`
(TIC-0100B1↔TCV-0100B1) is additionally a likely **answer-key/prompt mismatch**: labeled TRUE
because it's "the same ISA instrument loop," but the prompt asks about a *physical pipe or
line* — a loop relationship is a signal-convention association, not a traceable connecting
line, and no such line is visible in the crop. GPT-5.5 answering NO here is defensible, not a
model error.

**Corrected-box retest (3 of the 4 pairs re-boxed by hand around the actual drawn symbol —
vessel outline, tank rectangle, physical valve bowties — instead of the tag text; `pair_18`
dropped as mislabeled, `Benchmark_Gaps_Register.md` scope reduced to 18 clean pairs):**

| | Original boxes (18 clean pairs) | Corrected boxes (3 of 18 fixed) |
|---|---|---|
| GPT-5.5-low, short YES/NO | 12/17 = 70.6%* | 14/18 = 77.8% |
| GPT-5.5-low, CoT v2 | 13/17 = 76.5%* | **15/18 = 83.3% — clears the 80% bar** |

*recomputed from the 19-pair ceiling run, excluding the dropped `pair_18`.

Per-pair detail on the 3 corrected crops, read honestly rather than just the aggregate:
`pair_01` (vessel↔transformer) flipped to correct under CoT — real vertical line traced
correctly once the box was on the transformer symbol, not its tag. `pair_10` (tank↔H360
valve) stayed wrong on both prompts even with the better box — GPT traced the real pipe
correctly to a different endpoint (H358/drip tray) and correctly said the traced line does
NOT reach H360 as boxed, which may mean this specific answer-key entry deserves a second
look, not just an artifact fix. `pair_14` (two same-line valves) flipped correct under the
short prompt but the CoT reasoning described the line "passing through" the blue valve and
then answered NO anyway — a real reasoning-precision miss, not a box issue.

**Bottom line — three separate, now-disentangled findings:**
1. **Neither zero-shot Qwen3-VL-8B nor the v3-relation adapter can do bounded connectivity
   verification on real AG/RIVE sheets today.** Qwen's proven strength is reading (Probe 3),
   not visual line-tracing through clutter/crossings — a different skill.
2. **The hybrid architecture's premise is not dead, just unavailable locally right now.**
   With clean symbol-extent boxes (Probe 1's method, not tag-text boxes) and CoT prompting,
   GPT-5.5-low clears the 80% bar (83.3%). Box-placement quality was hiding a real chunk of
   its actual capability — this is n=3 corrected pairs, directional evidence, not a
   full-bundle re-benchmark.
3. **Recommended path:** proceed with the deterministic upgrades that need no model at all
   (process-backbone pass gap #14, claim partitioning gap #22, line-label endpoint
   filtering) — these are unaffected by any of the above and attack pipeline 3's measured
   weak strata directly. Do not sink further GPU time into v3-relation without first (a)
   finishing its remaining training epochs and (b) building a genuinely held-out AG/RIVE-
   domain relation eval — today's data says that investment isn't justified yet. If a
   verifier role is revisited later, it should use symbol-extent boxes (not tag-text boxes)
   and a model at least as capable as GPT-5.5-low — no currently-available local model has
   cleared this bar.

---

## Minimum credible benchmark vs full-fledged

- **Buildable now (Group 1 only):** topology-only, PID2Graph, raster-CV-only, 3 arms +
  floor baseline, stratified, cost-accounted. Credible for "which architecture finds real
  connections," silent on relation kinds, hierarchy, and vector-path advantage.
- **Full-fledged (Group 1 + Group 2 yeses):** adds real-PDF sheets (vector path tested),
  tag-attributed edges, hierarchy scoring, and kind-level scoring on the fixture subset.
  The delta is ~2–4 hours of human annotation + two decisions.

---

## Part D — Tier-1 upgrades wired end-to-end, R4 validator tested in-pipeline, real-sheet
## recall verification (2026-07-25)

Follow-on to Part C: the three buildable Tier-1 upgrades (#1 off-page partitioning, #2
process-backbone pass, #4 line-label endpoint filtering — #3 symbol-extent stays BLOCKED,
no change from Part C) got wired into the actual pipeline and run end-to-end, with both a
before/after edge-count capture and a real-sheet recall check — not just the isolated
self-tests Part C reported.

**Upgrade #2 (process-backbone pass) — corpus-validated before trusting it, then benchmarked
at real scale, not just the single self-test sheet:**

Corpus check across all 762 local PID2Graph sheets — "if R2a walks the line graph through one
inline valve/instrument/pump instead of always stopping at the first symbol reached, how often
is that a real GT edge vs. an invented false positive?"

| Fitting walked through | Candidate pairs | Already a real GT edge | Would be a new false positive |
|---|---|---|---|
| Valve | 324,495 | 84.6% | 15.4% |
| Instrumentation | 141,733 | 84.8% | 15.2% |
| Pump | 1,526 | 74.7% | 25.3% |

Single self-test sheet (OPEN100/0, R1+R2a) before/after: **F1 0.104 → 0.156**, precision
0.385 → 0.375 (−3%), recall 0.060 → 0.099 (+65%), TP 20 → 33, FP 32 → 55.

**Then the honest multi-sheet number (12-sheet real OPEN100 aggregate, R1+R2a):
F1 0.214 → 0.225** — smaller and messier than the single-sheet number suggested. Recall +16%,
precision −16%; false-positive growth (54%) came in well above what the corpus check predicted
(~15%), because the corpus check only validated the single-fitting-hop case, not the longer
chains the pass also produces walking through multiple fittings in sequence in practice.
**Dataset PID (the 500-sheet synthetic dense tree) was attempted and abandoned for this
benchmark:** R1's raster pipeline could not finish a single Dataset PID sheet in 5+ minutes —
those images are ~7168×4561px (7× OPEN100's pixel count) and R1 was ported without the
downscale step production's Stage 6 actually has. **Dense-sheet stratification remains
untested for the backbone pass** — the multi-sheet number above is sparse-sheets-only.

**Real-sheet end-to-end run.** New: `relationship_pipeline.py` — the full pipeline over given
entities (isolated mode), `upgraded=True/False` toggling original vs. upgraded R2a/R2b. Ran on
all 3 real AG/RIVE dev sheets, both configs (pre-LLM candidate edge counts):

| Sheet | original pre | upgraded pre |
|---|---|---|
| PX-2368-0180004-001 | 78 | 52 |
| GD-B-540-DP-2920-005-Z | 140 | 201 |
| PX-2365-0140006-001 | 106 | 150 |

**R4 (local Qwen3-VL-8B + v3-relation, in-pipeline) — confirmed degenerate, matches Probe 2
exactly.** `build_r4_bundle.py` rendered one crop per unique candidate edge (v3-relation's
trained crop format), pushed as `r4_bundle_2026-07-25.zip`
(`timthy45/pnid-extraction-datasets/benchmarks/`, now also mirrored in
`benchmarks/rescue_bundle_2026-07-25.zip` per the 2026-07-27 rescue). GPU notebook
(`notebooks/e2e_harness/Pipeline3_R4_Validation_Benchmark_GPUOnly.ipynb`) ran an 8-candidate/
sheet smoke gate before committing to the full 492-candidate run:

| Sheet | original pre | original post | upgraded pre | upgraded post |
|---|---|---|---|---|
| PX-2368-0180004-001 | 78 | **0** | 52 | **0** |
| GD-B-540-DP-2920-005-Z | 140 | **0** | 201 | **0** |
| PX-2365-0140006-001 | 106 | **0** | 150 | **0** |

R4 said "No" to every single candidate it saw, on every sheet, in both configs — it doesn't
filter, it deletes. This is the same degenerate collapse Probe 2 found (Part C), now reproduced
inside the actual pipeline stage, not just an isolated 19-crop probe. **The pre-LLM
deterministic result is therefore the real benchmark number** — exactly the fallback the
explicit pre/post split (a Tom design requirement: "incase its giving worse results we have the
result even without verification to show") was built for. Full 492-candidate run deliberately
NOT executed — see Group 2, gap #23, for the still-open decision on whether to run it anyway.
Results confirmed pushed to HF as `benchmarks/r4_validation_results_2026-07-25.json`
(`smoke_cap: 8`, verified 2026-07-27).

**Opus recall verification, PX-2368 only — a real, hand-traced number, not certified GT.**
26 real on-sheet connections across the sheet's 3 main equipment clusters (MBD-0100 surge
vessel, NBK-0300 treater + XFMR-0301, PBA-0201/0202 pumps) traced by eye from the raw drawing
and checked against what the deterministic pipeline found:

- **Recall — strict: 58% (15/26).** Crediting near-misses (wrong endpoint but real connection
  found): **77% (20/26).**
- **Recall is IDENTICAL in both configs** — the Tier-1 upgrades don't touch which real
  connections get found, only the false-positive composition. The 6 missed connections
  (SDV-0100F, SDV-0300A, and all 4 pump-suction instruments — FSV/PSHL on both LACT pumps) are
  missed the same way whether the backbone pass and line-filter are on or off.
- 5 more are "wrong endpoint" (e.g. LSHL-0100A/TSH-0100A bound to HAM instead of the vessel;
  PSV-0300A/FSV-0300C/PSV-0300C bound to XFMR/pumps instead of the treater).

**Note the OPEN100 recall gain above (+65% single-sheet, +16% multi-sheet) and this "recall
unchanged on PX-2368" finding are not in tension — they're two different measurements.** The
OPEN100 numbers are formal P/R/F1 against real PID2Graph edge ground truth; the PX-2368 number
is an informal hand-trace against 26 real connections on one AG/RIVE sheet, and the specific
misses there happen to be structurally out of the backbone pass's reach on that sheet (missing
symbols/instruments, not missing fitting-hops). Don't average these into one "does the backbone
pass help recall" answer — report both, scoped to what each actually measured.

**Delta adjudication, PX-2368 — every backbone-added edge + a sample of line-removed edges,
judged against the real drawing:**

| Upgrade effect | Verdict | Finding |
|---|---|---|
| Backbone-added edges (#2) | 0–1 of 5 correct | 4 of 5 falsely link two SEPARATE off-page connector labels stacked on the sheet border (`MBF-0623↔HBG-0905`, `PBA-0501↔PBA-0903`, `MBF-0500↔PBM-0450/0451`) — a NEW failure mode, not seen in the 762-sheet corpus check above, which only validated the single-fitting-hop case on real drawn geometry, not off-page label adjacency |
| Line-removed edges (#4) | Correctly spurious | Real precision win — dropped junk like `2"-245-PSIG↔2"-245-PSIG` (two pipe labels) and `HAM-0100↔6"-245-PSIG` (equipment↔a pipe label) |

**Bottom line:** the deterministic pipeline recovers ~58–77% of the real on-sheet connections
on the one sheet checked this way. The Tier-1 upgrades move false-positive composition, not
real-connection recall — the backbone pass trades away precision by inventing off-page links on
real sheets (a risk the corpus check didn't fully predict), the line-filter buys some back by
removing pipe-label junk. **The local Qwen R4 validator cannot help with any of this — it
rejects everything, confirmed twice now (Probe 2, then in-pipeline).** No model available
locally today can currently do this verifier role on real sheets (Part C's GPT-5.5-low ceiling
result, 83.3% with corrected boxes, is the only thing that's cleared the bar so far, and that
was on PID2Graph-style symbol-extent boxes, not yet retested on real AG/RIVE edges).

**Explicitly NOT done in Part D, flagged rather than silently skipped:**
- Recall-tracing the other 2 of 3 real sheets (GD-B-540, PX-2365) — only PX-2368 was traced.
- A GPT-5.5-low R4 arm — cheap, no GPU, would answer "does a competent validator help" as its
  own question, separate from the local-substitution question. Proposed, not built.
- Wiring the backbone pass into `agreement_diff.py`'s AG/RIVE path (Group 2, gap #25).
- Tightening the backbone pass to single-fitting hops only, given the longer-chain false-positive
  growth exceeded what the corpus check validated — proposed (my recommendation), not built.
