
# End-to-End Harness — Build Plan

**Date:** 2026-07-16
**Depends on:** `Conversion_Layer_Plan.md` (BUILT — `src/e2e_bench/converters`, `assembly`,
`backends`, all 25 tests passing). This plan builds ON TOP of that layer as a library; it
does not re-implement any schema mapping.

**Purpose:** score the real `pnid-intelligence-agent` pipeline two ways on the same input
sheets — **Arm P** (production-faithful: GPT-5.5-low at every LLM-driven stage) vs **Arm L**
(local: best model per stage, per this project's own benchmarking so far) — and use the gap
to decide what still needs fine-tuning, per the original ask ("one input P&ID sheet, runs
through the entire agent the way it's supposed to in production, verify the output, key
takeaways, decide how to go ahead with training").

**2026-07-16 update — this is a proof-of-concept, not the production benchmark.** Tom's
explicit framing: "I just need to show some results... not the version that will be going
to production, more like a proof of concept — if it works I'll be given more time to fine
tune and play around." This changes the priority order: **a real, defensible first number
beats exhaustive coverage.** Concretely:

- **Holdout decision: option (c) locked in** (§2) — proceed with the existing seed-disjoint
  PID2Graph split, caveat relation-set F1 as possibly in-sample for the v3-relation
  adapter specifically, don't block on retraining.
- **POC scope trim (supersedes some of §3-6 below until the POC lands a result):**
  - **Cascade mode only.** Isolated mode (stage 4 / stage 12 fed clean GT,
    §1 H1/H2) is real diagnostic value but not needed to show a first end-to-end number —
    defer to the "more time" phase if this POC lands.
  - **Arm L = Qwen everywhere, not Molmo2+Qwen mixed.** Qwen naturally emits `entity_type`
    + a tag/value together when prompted for it, unlike Molmo2 (points only). Using Qwen
    for stage 4 too means D-M1's tag-matching geometry (§4.3) isn't needed for the POC at
    all — it becomes a documented stretch comparison for later ("does swapping in Molmo2 +
    simple tag-matching beat Qwen's own detection+tagging"), not a blocker now.
  - **Small holdout, 3-5 sheets**, not 5-15 — enough to show a real number, cheap enough to
    debug fast if something breaks.
  - H2's clean-input false-positive-removal diagnostic (stage 13 on real GT entities) is
    cheap and worth keeping in scope even for the POC — it's nearly free once entity
    matching exists.

---

## 0. What's already decided, and where each number comes from

Every per-stage local candidate below is a REAL result already produced in this project —
this plan does not re-benchmark stages in isolation, it wires already-chosen winners into
one end-to-end run and measures compounding/interaction effects a per-stage score can't see.

| Stage | Production (Arm P) | Local (Arm L) | Status |
|---|---|---|---|
| 01 Sheet classification | GPT-5.5-low | Qwen3-VL zero-shot | Believed sufficient, never formally scored in isolation — cheap to verify (§4) |
| 01.5 Page OCR | Google Vision | PaddleOCR | **Decided, not yet benchmarked at all** (pipeline_status.html) — first real run happens as part of this harness |
| 02 Title block | GPT-5.5-low | Qwen3-VL + tuned prompt v2 | Real: 67% all-fields-correct, ties GPT-5.5-low (n=6) |
| 04 Symbol detection | GPT-5.5-low | Molmo2-O-7B (tile=512/upscale=2/enhance) | Real: F1=0.628, best local so far, still below 0.70 pass bar |
| 10.5 Skid grouping | GPT-5.5-low | Qwen3-VL zero-shot, per-symbol prompt | Real: 92.3% vs GPT-5.5-low's 91.9% on constructed ground truth (n=12) — best local result in the project |
| 13 Entity validation | GPT-5.5-low | Qwen3-VL + v3-stage13 LoRA | Real: 89.2% vs 66.7% (n=120) |
| 12 Relation validation | GPT-5.5-low | Qwen3-VL + v3-relation LoRA | Real: 89.2%* vs 72.5-80% (n=120, *only 1/3 training epochs done) |
| 00/03/06/11 | deterministic, no model | same code, both arms | N/A |

**Open, load-bearing design decision this plan must resolve (Conversion_Layer_Plan.md §11's
biggest finding):** a detection needs a non-empty `value` (tag text) or `build_entity`
silently drops it. Molmo2 emits points only, no text. **Decision (D-M1, see §3):** pair
Molmo2 with a cheap, deterministic nearest-OCR-word tag-matching step (pure geometry, no
model call) so its detections can carry a `value` and actually participate past stage 4.
This is itself something to score (does simple proximity-matching produce good enough tags,
or does it need to be smarter) — report it as its own finding, not hide it inside the
detection number.

---

## 1. Decision log

| # | Decision | Rationale |
|---|---|---|
| H1 | Two execution modes: **cascade** (each stage consumes the SAME arm's real prior-stage output) and **isolated** (stage fed real ground truth instead of prior-stage output). Isolated mode is only built for the stages where PID2Graph provides real ground truth at that exact boundary: **detection (04)**, **relation validation (12)**. Title block and skid grouping have NO isolated mode — no dataset has that ground truth (confirmed by direct inspection, twice, already) — they're scored only via their existing standalone benchmarks and simply plugged into the cascade run using their best already-established config. | The whole point of isolated mode is per-stage error attribution ("is the compounding loss from stage 4 or stage 13?") — only meaningful where a clean, real reference input exists to isolate with |
| H2 | Entity validation (13) gets a THIRD scoring view beyond cascade/isolated: **clean-input false-positive-removal rate** — feed it the real GT entities (all genuinely real) and measure what fraction it wrongly removes. This is different from "isolated mode" in the detection/relation sense (there's no upstream noise to isolate FROM — GT entities have no detection step) but answers the same kind of question: does stage 13 have an inherent false-rejection bias, independent of what stage 4 hands it? | Directly reuses PID2Graph's real entities without needing a detection step at all — cheap, high-value diagnostic |
| H3 | Test data: PID2Graph file-disjoint holdout (§2) for the full detect→relate graph score; the existing frozen Gupta 20 test sheets for a detection-only + title-block side-check (no relation GT there, already established) | Matches what real ground truth actually supports; doesn't pretend Gupta has relations it doesn't |
| H4 | D-M1 (Molmo2 tag-matching): nearest-OCR-word-within-radius, radius = a fraction of the detection bbox diagonal (tune empirically, start at 1.5×), assign that word's text as `value`. If no word within radius, `value=None` and the entity is (correctly, per agent behavior) dropped — counted and reported, not hidden. | Cheapest possible fix that unlocks Molmo2 for anything past stage 4; explicitly NOT claiming this is as good as a real tag-reading step |
| H5 | Metrics reported separately per this project's own rule (never average detection+typing into one number, extended here to never average entity-quality and relation-quality): **entity-set F1** (match by type + bbox IoU ≥ 0.5 against PID2Graph GT nodes) and **relation-set F1** (match by (source,target,name) against PID2Graph GT edges) as two separate headline numbers per arm/mode, never blended | Direct extension of the two-part-metric rule already governing this whole project |
| H6 | Compute split (real constraint, not mechanical rule-following — see §5 for why cascade mode breaks the clean CPU/GPU separation): isolated-mode Arm L inference is fully batchable ahead of time in Colab (GPU-only, `runtime.unassign()` at the end). Cascade-mode Arm L must run GPU inference AND the agent's deterministic CPU code (tiling, line tracing, graph construction, the conversion layer) interleaved per stage-transition within ONE Colab session, because stage N's real input depends on stage N-1's real output for that SAME arm — there's no way to pre-batch that. Arm P (GPT-5.5, API-only) runs entirely on the Mac in both modes, no GPU needed ever. | Honest architecture given the actual data dependency, not a forced fit to the general CPU/GPU rule |
| H7 | Graph-matching (entity/relation F1) needs a bbox-IoU-based + name-based matcher between the agent's `BundleEntity`/`BundleRelation` output and PID2Graph's raw graphml nodes/edges — this doesn't exist yet and must be built as part of this harness (§4), it's not part of the conversion layer (that layer goes INTO the agent's schemas, this matcher goes FROM the agent's schemas back to PID2Graph's for scoring) | Scope boundary: conversion layer = model→agent; this matcher = agent→ground-truth |

---

## 2. Test data — file-disjoint PID2Graph holdout

**RESOLVED (2026-07-16) — a genuinely file-disjoint holdout is NOT achievable against the
already-trained v3-relation adapter, for either PID2Graph tree.** Confirmed by direct
investigation of the training notebook's actual data-selection code, not assumption:

- Training (`Stage4_PerStage_Stage13_and_Relation_v3.ipynb`, `build_relation_examples_from_
  tree`) walks `sorted(Patched_tree.rglob("*.graphml"))` — **the same `Patched` tree** the
  benchmark eval code also walks — shuffles with `random.Random(777)`, and slices the first
  1,600 (OPEN100) / 4,000 (Dataset PID) patches. This is patch-FLAT sampling, not
  sheet-stratified. No frozen split file exists for PID2Graph (unlike Gupta's
  `test_ids.json`).
- Confirmed patch counts (local extraction): OPEN100 = 1,629 total patches across 12
  sheets (~136/sheet); Dataset PID = 19,462 total patches across 500 sheets (~39/sheet).
  Training used 1,600/1,629 (98.2%) of OPEN100 and 4,000/19,462 (20.6%) of Dataset PID.
- **Expected number of sheets with ZERO patches touched by training:** OPEN100 ≈ 0.0 out
  of 12; Dataset PID ≈ 0.065 out of 500 (computed as `(1 - patches_used/total_patches) **
  patches_per_sheet`, per sheet, summed). Both are effectively zero — **training almost
  certainly touched every sheet in both trees at least once.**
- Also: `Patched`'s per-subfolder bare filenames collide heavily across sheets (the same
  bug class already hit once this project for skid grouping) — the real disjointness unit
  has to be the SOURCE SHEET (subfolder), not the bare filename, and patches from the same
  sheet are highly correlated (adjacent tiles of one drawing) — so even "different patch,
  same sheet" would still be leakage, making the problem worse than a naive file-count
  check would suggest.

**Three real options, not a guess among them — this needs Tom's call before proceeding:**

| Option | What it means | Cost |
|---|---|---|
| (a) Different data source | Evaluate relation-stage quality on some dataset other than PID2Graph | No other real dataset with both symbols+edges is known to exist in this project (already searched exhaustively) — likely blocked without new data acquisition |
| (b) Retrain with a frozen sheet-level holdout | Freeze N sheets per tree BEFORE any training touches them (real, separate effort — mirrors what should have been done for Gupta's `test_ids.json` from the start), then retrain v3-relation from scratch against that holdout | Real GPU time + delays the harness; the methodologically correct fix |
| (c) Accept the seed-disjoint split as a documented limitation | Proceed now with the existing seed-8181-vs-777 split, label every relation-validation result (cascade AND isolated) with an explicit "may be in-sample for this specific adapter" caveat | Fastest; **asymmetric between arms** — Arm L's v3-relation adapter may have partial familiarity with holdout sheets, Arm P (GPT-5.5) has none, so any Arm-L-beats-Arm-P relation-F1 result is weaker evidence than it looks |

**Recommendation: option (c) for now, loudly caveated, with (b) flagged as the real
priority fix before this becomes a production decision rather than a research direction.**
Reasoning: entity-set F1 (stage 4 detection scoring) is NOT affected by this at all — Molmo2/
Qwen's stage-4 config was never trained on PID2Graph — only relation-set F1 specifically
(evaluating the v3-relation adapter) inherits the leakage risk. Blocking the entire harness
on (b) delays a benchmark that's still valuable for everything except that one number.

**Holdout size, independent of the above:** realistically 5-15 sheets — each sheet is dozens
of model calls per arm per mode, so this stays well below n≥100 regardless; the deliverable
is end-to-end score + per-stage attribution, not a decision-grade single number, same
caveat this project applies everywhere small-n shows up. Freeze the chosen sheet IDs in
`e2e_holdout_ids.json` (mirroring Gupta's `test_ids.json` discipline) the moment they're
picked, regardless of which option above is chosen.

---

## 3. Package layout

```
pid-ml/src/e2e_harness/
  __init__.py
  holdout.py              # loads e2e_holdout_ids.json, asserts disjointness from training
  ground_truth.py          # PID2Graph graphml -> GT entities/edges (reuses parse_graphml
                           # pattern from synth_skid.py, but WITHOUT the equipment-only
                           # filter this time — GT needs every real node type for fair
                           # entity-F1 scoring, matching whatever the ontology maps to)
  graph_matcher.py         # BundleEntity/BundleRelation vs GT nodes/edges -> entity F1,
                           # relation F1 (H5, H7)
  tag_matching.py          # D-M1: nearest-OCR-word matcher for Molmo2 detections
  arms/
    __init__.py
    arm_p_gpt55.py         # Arm P: GPT-5.5-low at every LLM stage, runs entirely on Mac
    arm_l_qwen.py          # Arm L "cascade": Qwen (+ adapters where established) at every
                           # LLM stage - the GPU+agent-code interleaved Colab notebook (H6)
    arm_l_molmo_detection.py  # Arm L detection sub-config: Molmo2 + tag_matching.py,
                           # swappable into arm_l_qwen's stage-4 slot for comparison
  isolated/
    __init__.py
    isolated_detection.py    # stage 4 scored directly against PID2Graph GT (this
                             # basically IS the existing detection benchmark methodology,
                             # just on the new holdout)
    isolated_relation.py     # stage 12 fed PID2Graph's real edges as "already-proposed
                             # relations needing validation", scored on confirm/reject
                             # accuracy against the fact that they're all real (should
                             # mostly confirm)
    isolated_entity_clean.py # H2: stage 13 fed real GT entities directly, measures
                             # false-positive-removal rate
  reporting/
    __init__.py
    scorecard.py             # assembles the final per-arm/per-mode/per-stage report,
                             # pushes to HF, appends to pid-ml/results.csv
  tests/
    fixtures/
    test_graph_matcher.py
    test_tag_matching.py
    test_holdout.py
```

---

## 4. The pieces that don't exist yet (this plan's actual work)

### 4.1 `ground_truth.py` — PID2Graph GT extraction

Reuses the `parse_graphml` pattern already built twice this project (synth_skid.py,
the earlier skid-matrix work) — this time WITHOUT the equipment-only node filter (that
filter existed because skid membership is meaningless for `connector`/`crossing` nodes;
entity-F1 scoring needs the real full node set so precision/recall reflect the whole
graph, not a pre-filtered subset). Output: `GTEntity(node_id, bbox, label)` and
`GTEdge(source_node_id, target_node_id, line_type)` lists per sheet, from the `Complete`
tree (full sheets, not `Patched` — the Patched-tree bugs found earlier this project, both
the sheet-id collisions and the arbitrary-crop-boundary problem, apply here too).

### 4.2 `graph_matcher.py` — the actual new scoring code (H7)

- **Entity matching:** greedy bipartite match between agent `BundleEntity.source_bbox`
  (agent/page coords) and GT node bbox (PID2Graph coords — **confirm these are the same
  coordinate space before matching**; if PID2Graph graphml coords aren't page-raster pixels
  matching what the agent's pipeline produces, a coordinate transform is needed first,
  check this before writing the matcher, don't assume). Match if IoU ≥ 0.5 AND
  `entity_type` maps to the same GT `label` under a fixed type-name mapping (build this
  mapping explicitly — agent entity types are the benchmark ontology's
  `valve/instrumentation/pump/tank/general/inlet_outlet/asset`; GT labels are PID2Graph's
  `valve/instrumentation/pump/tank/general/inlet-outlet/connector/crossing/arrow/
  background` — decide explicitly whether `connector`/`crossing`/`arrow`/`background` GT
  nodes count against precision when an agent entity has no equivalent, or are excluded
  from GT entirely for this score, matching the skid-grouping precedent of treating them
  as non-equipment).
- **Relation matching:** match agent `BundleRelation(source_temp_id, target_temp_id)` →
  (via the entity match above) → GT node pair, against GT edges by node-pair membership
  (undirected, since PID2Graph edges don't have forward/reverse semantics the way
  BundleRelation does — confirm this against Agent_Pipeline_Facts.md / the earlier
  build_relation_pool code, which treated `frozenset((a,b))` as the edge identity).
- Standard precision/recall/F1 from the match counts. Report per-sheet AND aggregate
  micro-F1 (matching the Stage 4 detection notebook's convention:
  `P = matched/pred, R = matched/gt, F1 = 2PR/(P+R)`).

### 4.3 `tag_matching.py` — D-M1

Pure geometry, no model call: for each Molmo2 detection (bbox, post-tile_to_drawing
projection), find OCR words (from the SAME page's stage-01.5 output) whose bbox center
falls within `radius = diagonal(detection.bbox) * 1.5` of the detection's bbox center;
assign the nearest such word's text as `value`. No match → `value=None` (dropped
downstream, counted). Report the match rate itself as a finding (what fraction of Molmo2
detections got a tag at all) — this number IS part of the result, not an implementation
detail to hide.

### 4.4 Arm implementations

**Arm P (`arm_p_gpt55.py`):** runs entirely on the Mac (`.venv-e2e`, same environment
Conversion_Layer_Plan.md's tests already run in). Per holdout sheet: call GPT-5.5-low with
each stage's REAL prompt (reuse the exact prompts already validated in this project's
per-stage benchmark notebooks — title block's, the skid per-symbol prompt, stage13/12's
existing tool-schema-based prompts from the original PerStageV3 notebook) → normalize via
`e2e_bench.backends.parse_gpt_json` → convert via `e2e_bench.converters.*` → deterministic
agent stages run for real in between. Cascade and isolated modes both fit on the Mac for
this arm since there's no GPU dependency anywhere in it.

**Arm L cascade (`arm_l_qwen.py`):** the Colab notebook (H6). Loads Qwen3-VL base + the
three existing adapters once (same pattern as `Stage105_SkidMatrix_Molmo2_Qwen_Adapters_
GPUOnly.ipynb`), AND has `pnid_agent`/`rive_adk`/`entity_operations` installed in the SAME
Colab environment (this is the one place this project's "new notebooks split CPU-prep
local / GPU-only Colab" rule gets a deliberate, documented exception — call this out
explicitly to Tom before building it, per that rule's own "flag before starting" clause).
Per holdout sheet, per stage: run the local model with its established best config
(Qwen zero-shot for 01, PaddleOCR — CPU, can actually run in this same notebook trivially
— for 01.5, Qwen+tuned-prompt for 02, Molmo2+tag_matching OR Qwen-adapted for 04 [both
configs run, compared], Qwen zero-shot per-symbol for 10.5, Qwen+v3-stage13 for 13,
Qwen+v3-relation for 12) → normalize → convert → run the REAL deterministic agent stage →
feed forward to the next LLM stage. `runtime.unassign()` at the end.

**Isolated-mode Arm L:** fully batchable in Colab ahead of time (H6) — no interleaving with
agent code needed, since the "input" is fixed ground truth, not a prior real stage's
output. Batch-generate all isolated-mode model answers for the holdout, push raw answers
to HF, then run conversion + scoring entirely on the Mac (isolated mode's SCORING doesn't
need the full agent pipeline either — e.g. stage-4-isolated scoring is just the graph
matcher against raw detections, no stage 6/11 needed at all).

---

## 5. Why cascade mode can't follow the usual CPU/GPU split (H6, expanded)

This project's standing rule for new notebooks (established mid-session, see memory
`feedback_gpu_cpu_split`) is CPU-prep local + GPU-only Colab. Cascade mode genuinely
doesn't fit that shape: stage 13's real input is stage 11's real output, which depends on
stage 6's real output, which depends on stage 4's real output — each transition requires
the agent's actual Python code (not just data movement) to run BETWEEN two GPU-inference
calls for the same sheet. Splitting this across two machines means either (a) a live
per-stage round-trip between Mac and Colab (slow, fragile, and defeats "no unnecessary GPU
idle time" since Colab would sit waiting on Mac-side agent code between calls), or (b)
accepting that cascade-mode Arm L is one self-contained Colab notebook with both the GPU
models and the CPU agent code installed together. Option (b) is what H6 chooses — flag
this explicitly as a deliberate exception, not a silent violation of the standing rule.

---

## 6. Metrics & report shape (H5)

For each of {Arm P, Arm L} × {cascade, isolated-where-applicable}:
- **Entity-set F1** (vs PID2Graph GT nodes)
- **Relation-set F1** (vs PID2Graph GT edges)
- Stage 13's clean-input false-positive-removal rate (H2, arm-independent of detection
  noise)
- Molmo2's tag-match rate (D-M1, Arm L detection-sub-config only)
- Per-stage parse-failure counts (D10 from the conversion layer — never silently absorbed)

**The actual deliverable, per the original ask:** cascade-mode entity/relation F1 gap
between Arm P and Arm L, cross-referenced against isolated-mode stage 4 and stage 12
scores, tells you WHERE the gap comes from — if isolated stage 4 F1 is close between arms
but cascade entity F1 diverges a lot, the loss is compounding/propagation, not raw
per-stage capability, and points at different fixes (e.g. better NMS/dedup tuning) than if
isolated scores themselves are far apart (which points at needing more fine-tuning on that
specific stage, e.g. finishing v3-relation's remaining training epochs).

---

## 7. Open items for whoever executes this plan

1. **Section 2 (holdout) is not finalized** — do not start building against a specific file
   list until that's resolved and written in.
2. Confirm PID2Graph graphml coordinate space actually matches the agent's page-raster
   pixel space before writing `graph_matcher.py` (§4.2) — don't assume.
3. Decide the GT-vs-agent entity-type mapping for `connector`/`crossing`/`arrow`/
   `background` nodes (§4.2) — affects precision meaningfully given these are ~75% of
   PID2Graph's nodes.
4. Pull the exact stage-13/stage-12 prompts + tool schemas from the original
   `PerStageV3_Stage13_Relation_vs_GPT55.ipynb` notebook for Arm P's GPT-5.5 calls (Arm P
   must use the SAME prompts already validated to work, not new ones) — verify they still
   match after any interim edits to that notebook.
5. Time-box `stage_11_run`'s full config surface (Conversion_Layer_Plan.md PHASE0_REPORT
   already confirmed it's importable and its 38 params are default-heavy; confirm running
   it end-to-end, not just importing it, works as smoothly as stage 6/13/12 did) before
   committing to using it wholesale versus hand-chaining `build_entity`/`build_relations`/
   `infer_from_skid_groups` as done in the conversion layer's own smoke test.
6. Decide realistic holdout size given per-sheet cost (dozens of model calls × 2 arms ×
   up to 2 modes) — this is a real time/cost tradeoff to make explicitly with Tom, not
   silently default to "as many as possible" or "5 because that's easy."


