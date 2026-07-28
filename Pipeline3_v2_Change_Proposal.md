# Pipeline 3 v2 — symbol-first input, boundary ports, dual-mode scoring

**Status: PROPOSAL. Nothing here is built.** Written 2026-07-27 from the brainstorm on
switching Pipeline 3's input from "a list of all tags" to "coordinates + tags of the symbols."

Every claim below is marked with its evidential status:
- **[MEASURED]** — a real number from a real run in this repo
- **[PREDICTED]** — a falsifiable consequence of the design, with the number it should move
- **[HYPOTHESIS]** — plausible, untested, must not be assumed
- **[OPEN]** — needs a decision from Tom before it can be built

Read alongside `Benchmark_Gaps_Register.md` (Part C/D + Group 2 gaps 23-26) and `HANDOFF.md`.

---

## 1. Why change anything

Pipeline 3 currently consumes prod's OCR/Vision tag list, where each entity's `bbox_px` is the
location of its printed **text label**, not its **drawn symbol**. This single confusion is the
most recurring bug source in the project:

- **[MEASURED]** MBD-0100's tag bbox is a 130×25px text label sitting apart from its drawn
  ellipse — no traced pipe endpoint ever resolved to it. Root cause of the original near-zero
  agreement result.
- **[MEASURED]** Probe 2's crops boxed tag text on 4 pairs every model missed. Hand-correcting
  3 boxes moved GPT-5.5-low from 76.5% → **83.3%**, clearing the 80% bar it had been failing.
- **[MEASURED]** The 2026-07-24 rerun gave off-page equipment mentions narrow ~110×20px
  label-shaped bboxes — the "0 off-page claims" anomaly.
- **[MEASURED]** The mitigation (`_inflate_equipment_bboxes`, `EQUIPMENT_BBOX_PAD_INCHES`) is
  partial and insufficient: MBD-0100 gained 1 connection, GD-B-540 stayed at exactly 0,
  PX-2365 went 0→2.
- **[MEASURED]** Tier-1 #3 (resolve real symbol extent from vector geometry) is BLOCKED:
  `closePath` is `False` on all 5,608 tested paths; path bboxes span sub-pixel glyphs to
  full-page rects; no signal separates equipment outline / pipe / text stroke.

### The unlock

Two of our own results look contradictory until lined up:

| Test | Verdict |
|---|---|
| Probe 1 — symbol-extent reconstruction from vector geometry | **PASS** (incl. branching case) |
| Tier-1 #3 — symbol-extent resolution | **BLOCKED** |

Same problem, opposite outcomes. The difference: **Probe 1 was seeded** — it knew roughly where
to look. #3 was unseeded: pick the right path out of 5,608 with no starting hint.

**Molmo2's point is that seed.** #3 stops being blocked-pending-design and becomes
implementable, using the method Probe 1 already validated. That is the core of this proposal.

Supporting: **[MEASURED]** Molmo2 is the best available localizer we have — frozen 20-sheet
Gupta test set, class-agnostic detection: Molmo2-points **F1 0.6276** (P 0.6309 / R 0.6244) vs
GPT-5.5-low **F1 0.5125** (P 0.5335 / R 0.4932). Frontier models are *worse* at this specific
task; Molmo2 has a native pointing head while the others emit coordinates as text tokens.

---

## 1.5 Diagnostic run 2026-07-27 — five new measured facts

Run directly against `benchmarks/extraction_2026-07-24/PX-2368-0180004-001_gpt55low.json`
(104 tags). These change the priority order, so they're recorded before the design.

**[MEASURED] 1. All 6 MISSED connections are extraction *absences*, not tracing failures.**
- `SDV-0100F` — **absent** from the tag list entirely.
- `SDV-0300A` — **absent**.
- The 4 pump-suction instruments — **absent**. Only 12 unique instrument/safety tags exist on
  the whole sheet (all 0100/0300-series); a spatial sweep within 900px of the LACT pump pair
  (`PBA-0201`/`PBA-0202`) returns only the pumps themselves, `PSV-0300A`, `FSV-0300B`,
  `PSV-0300C` and one line label — no dedicated pump-suction instruments at all.

→ **Pipeline 3's relation logic is not the binding constraint on those 6.** It never received
the entities. Extraction recall is. This is the strongest available argument for the Molmo2
retraining work.

**[MEASURED] 2. Equipment bboxes are text labels — all 31 of them.** Median bbox by type:

| type | n | median w×h | reading |
|---|---|---|---|
| equipment | 31 | **120×20 px** | unambiguously a text label, not a drawn shape |
| line | 45 | 144×20 px | correct — it *is* text |
| safety_device | 8 | 106×30 px | label-shaped |
| instrument | 8 | 58×42 px | plausibly the ISA bubble — roughly correct |
| valve | 12 | 58×42 px | plausibly the symbol — roughly correct |

The MBD-0100 finding (130×25px) is not an outlier; it is the rule for every equipment entity.

**[MEASURED] 3. The failure mode is specifically an *equipment-endpoint* failure.** All 5
WRONG-ENDPOINT cases have a wrong **equipment** end, never a wrong instrument end:
`LSHL-0100A`/`TSH-0100A` → HAM instead of the vessel; `PSV-0300A`/`FSV-0300C`/`PSV-0300C` →
XFMR/pumps instead of the treater. Consistent with fact 2: instrument bubbles are boxed roughly
right (58×42), equipment is boxed as text (120×20), so a line leaving a correctly-placed
instrument resolves to whichever equipment *label* happens to be nearest — often the wrong one.

→ **Scoping consequence:** extent resolution matters most for **equipment**. An equipment-only
v1 should capture most of the win, and shrinks the Phase 1 A/B from ~30-35 hand-placed extents
to roughly **6-8** (vessel, treater, XFMR, 2 pumps, HAM).

**[MEASURED] 4. Prod's extraction output carries zero symbol-shape signal.** `symbols: []`,
`elements: []`, `edges: []` all empty; `symbol_shape == "none"` on all 104 tags. The schema has
the fields and never populates them (matches the register's note that `edges` exists but is
never filled).

→ **v2's contract cannot be populated from prod output at all.** Symbol extents must come from
Molmo2 or from hand annotation. There is no cheap prod-based shortcut, and Phase 1 therefore
*must* use hand-placed extents.

**[MEASURED] 5. 43% of the "entity list" isn't entities.** 45 of 104 tags are `type: line` —
pipe labels. `LabelAnnotation` removes them from the node set by construction, which quantifies
what Tier-1 #4 was doing: filtering the single largest category.

**[HYPOTHESIS, new and cheap] ISA tag-series cross-check as a second signal.** All 3 mis-bound
safety devices are 0300-series (belonging to `NBK-0300`, the treater) but were bound to
0200-series pumps. Tag numbering already encodes the right answer, independent of geometry. A
deterministic guard — "warn/reject when an endpoint's ISA series disagrees with its resolved
equipment's series" — could catch these mis-bindings *even without* perfect extents. Essentially
prod's existing free loop-grouping prior applied to endpoint validation.

---

## 2. New input contract

The entity-extraction stage stops emitting "all tags" and emits three distinct things. **The
text layer does not go away** — it is demoted from *the entity list* to *an attribute source*,
feeding tag association, port reference text, and line specs.

```python
@dataclass
class SymbolNode:
    id: str
    point: tuple[float, float]            # seed, page-raster coords
    extent: BBox | None                   # resolved drawn-shape bbox
    extent_source: str                    # vector_seeded | raster_seeded | radius_fallback | hand
    extent_conf: float | None
    tag: str | None                       # None => untagged symbol, STILL A NODE
    tag_source: str | None                # nearest_ocr | vlm_read | hand
    tag_dist_px: float | None             # for auditing mispairs
    type: str | None                      # via type_vocab.py
    source_arm: str                       # which extraction arm produced it

@dataclass
class PortNode:                           # off-page connector — NEW node kind
    id: str
    extent: BBox                          # the connector graphic's real extent ON THIS SHEET
    ref_text: str | None                  # "FROM GLYCOL CONDENSATE SEPARATOR (MBD-0635)"
    ref_tag: str | None                   # parsed: "MBD-0635"
    ref_sheet: str | None                 # parsed sheet number
    direction: str | None                 # in | out | None, from the arrow
    ref_conf: float | None

@dataclass
class LabelAnnotation:                    # NOT a node, NOT a tracer endpoint
    id: str
    bbox: BBox
    text: str
    kind: str                             # line_spec | note | titleblock | unknown
```

**Invariants to enforce in code, not convention:**
1. Ports are **terminal** — never walked through.
2. Labels are never graph nodes and never tracer endpoints.
3. Untagged symbols **are** nodes (see §3.3).

### 2.1 The "no symbol" taxonomy

| Case | Treatment |
|---|---|
| Off-page connector | `PortNode` — real extent, terminal, carries remote reference |
| Line/pipe spec label (`6"(300#)`, `2"-245-PSIG`) | `LabelAnnotation` |
| Notes / flags | `LabelAnnotation(kind=note)`, ignored by graph |
| Title block | Out of scope (handled at Stage 2) |
| Instrument loop partners (TIC/TCV) | ISA tag-naming relation, no geometry, unchanged |
| Symbol found, tag unreadable | `SymbolNode(tag=None)` — **kept** |
| Tag named but no symbol anywhere on sheet | Dropped, or port-like reference if border-adjacent |

---

## 3. Stage-by-stage changes

### 3.0 R0 — entity ingest + extent resolution (NEW STAGE)

1. Ingest `symbols` / `ports` / `labels` from the configured source (§5 dual mode).
2. **Point → extent**, seeded:
   - *Vector path:* candidate paths whose bbox contains the seed → filter by plausible size band
     (reject full-page rects, reject sub-pixel glyphs) → smallest plausible enclosing shape.
     Probe 1's validated method.
   - *Raster fallback:* connected-component / flood-fill grown from the seed.
   - *Last resort:* fixed-radius box, explicitly flagged `radius_fallback`.
3. Emit `extent_source` + `extent_conf` on every node so **every downstream number can be
   stratified by extent quality.** Non-negotiable: this is what prevents R0 becoming an
   unmeasured black box in the middle of the pipeline.
4. Sanity asserts: extent contains its seed; area within a plausible band; cross-tile
   overlapping-extent dedup (Molmo2 ran tiled at 512px/2×; Stage 3 tiling is 1024px/205px
   overlap — stitching and dedup are required, not optional).

**R0 must be independently scoreable:** extent IoU against hand-verified extents. If R0 is
wrong, everything downstream is measuring R0's error, not relation logic.

### 3.1 Port detection — the honest gap

**[OPEN]** Where ports come from is *not* settled, and I want to be careful not to overclaim
Probe 3 here. **Probe 3 [MEASURED] 87.5% (7/8) validated Qwen *reading* connector text from a
given crop. It did not validate *finding* the connectors.** That detection step is unbuilt and
unvalidated.

Proposed v1 (cheapest, deterministic): off-page connectors sit at the sheet border by drafting
convention. We already have `diagram_area_bbox` and `bbox_center_outside` vendored standalone
from `isa_rules.py` — use them to isolate border-region annotations, then Qwen reads each one
(Probe 3's proven task). Alternative: train Molmo2 to point at connector graphics as a class.

### 3.2 R1 — line tracing

- Code largely unchanged; the **endpoint-resolution map** changes: resolve against symbol
  extents + port extents instead of text-label bboxes. **This is where the win lands.**
- Masking changes: mask *symbol extents* (correct) instead of OCR text boxes. **[MEASURED]** the
  OCR-text-masking experiment made things *worse* (F1 0.012 vs 0.023) because it removed real
  line content.
- **Delete** `_inflate_equipment_bboxes` / `EQUIPMENT_BBOX_PAD_INCHES` from the AG/RIVE path.
- **[HYPOTHESIS]** correct masking may lift R1's recall ceiling. **Must not be assumed** — the
  PID2Graph ceiling was root-caused to a corpus *rendering* mismatch, which better entities do
  not fix. Test separately; do not bundle into the headline A/B.

### 3.3 R2a — deterministic connectivity

- Passthrough becomes extent-based, keyed on resolved type.
- **Ports explicitly NON-passthrough.** **[MEASURED]** this kills the backbone pass's worst
  failure by construction: 4 of 5 backbone-added edges on PX-2368 falsely linked two *separate*
  off-page connector labels stacked on the border (`MBF-0623↔HBG-0905`, `PBA-0501↔PBA-0903`,
  `MBF-0500↔PBM-0450/0451`).
- Tighten backbone to **single-fitting hops**. **[MEASURED]** FP growth was 54% against the
  corpus check's predicted ~15%, because that check only validated the single-hop case, not the
  longer chains the pass produces in practice.
- **Do not kill the backbone pass yet.** Its verdict is currently "0–1 of 5 correct," but 4/5 of
  those failures were the off-page conflation that ports fix. **Re-adjudicate after the port
  change** rather than deciding now.
- Untagged symbol nodes participate normally — class-agnostic topology works without names, and
  this is consistent with Gupta and PID2Graph both being class-agnostic (CLAUDE.md rule 5).
  **[PREDICTED]** should *help* recall: today a symbol with no readable tag isn't a node at all,
  so traced lines dead-end into nothing.
- Tier-1 #4 (line-label filtering) becomes redundant by construction → **demote to a defensive
  assert.** If a label ever appears as an endpoint, that's a bug signal worth failing loudly on.

### 3.4 R2b — hierarchy

**[PREDICTED]** material improvement. **[MEASURED]** the containment prior currently fires
**zero** times (13/93 nodes parented, all via connectivity-nearest) because text-label bboxes
don't nest. Real symbol extents *do* nest — instrument bubbles inside vessel outlines,
instruments inside dashed skid boundaries. Cycle-break port stays verbatim.

### 3.5 R3 — entity validation

Largely unchanged. **[OPEN]** crops now box real symbols, but the v3-stage13 adapter's
**[MEASURED]** 89.2% was measured on PID2Graph crops — transfer to symbol-extent crops on real
sheets is unknown, and Probe 2 already showed v3-relation's PID2Graph number did *not* transfer
to AG/RIVE.

### 3.6 R4 — relation validation

**[MEASURED]** still degenerate: kept 0/8 on all 3 sheets, both configs; matches Probe 2's
52.6% chance-level result. Worth **one** cheap retest on real symbol-extent crops (we never ran
corrected-box Qwen — the corrected-box run was GPT-5.5-low only). **Low prior**: a degenerate
constant `"No."` is not a box-sensitive failure mode.

**Preserve the pre/post-LLM split** — explicit Tom design requirement, not an implementation
detail.

### 3.7 R5 — semantic enrichment

Better grounded. New: `LabelAnnotation(kind=line_spec)` gives pipe spec text, a possible
substitute for the line-typing **[MEASURED]** PR #711 proved dead from vector strokes (CAD
exports flatten to solid single-width strokes, 0% dashed).

---

## 4. agreement_diff.py — off-page becomes measurable

**[MEASURED]** Gap #22: of ~25 true GPT-5.5-low claims on PX-2368, ~19–20 had their remote
endpoint drawn only as border-annotation text, not as an on-sheet symbol. Only ~6 had both
endpoints physically drawn. Those 19–20 were declared structurally unverifiable.

With ports, `MBD-0100 feeds MBD-0635` decomposes into two answerable questions:
1. Does a traced path reach `port_k`? — deterministic geometry.
2. Does `port_k.ref_tag == MBD-0635`? — Probe 3's task, **[MEASURED]** 87.5%.

**[PREDICTED]** this converts ~19–20 previously-unscoreable claims per sheet into scoreable
ones, and turns gap #22's partition from a data-loss disclosure into a real metric. It also
splits work along each model's proven capability: **Qwen reads** (Probe 3 PASS), **geometry
traces** — neither is asked to do the other's job, which is exactly where Qwen failed (Probe 2,
hallucinated traces through clutter).

Bonus: connector arrows carry flow direction, which could partly restore directed scoring
(currently frozen undirected per gap #16 because GT direction was unreliable).

---

## 5. Scoring — dual mode, and the discontinuity guard

**The problem:** every existing Pipeline 3 number (**[MEASURED]** F1 0.214→0.225 across 12
OPEN100 sheets, plus all strata) used **GT-injected** entities via `_tags_from_gt_nodes`.
Switching the input to detected symbols compounds detection error with tracing error and makes
the result **incomparable to every number we have.**

**Guard: keep all three input modes permanently, tagged on every row, never mixed.**

| Mode | Entity source | Purpose |
|---|---|---|
| `gt_injected` | PID2Graph GT nodes | Isolates relation quality. Regression baseline. Free, already real GT. |
| `hand_verified` | Human-corrected extents | Certified reference on AG/RIVE, where no entity GT exists |
| `detected` | Molmo2 / multi-arm union | The honest end-to-end number |

This is the same principle as the pre/post-LLM split, on a new axis.

**Note:** the discontinuity is **already solved for PID2Graph** — GT entities are free there.
The gap is *only* AG/RIVE, which is Group 2 gap #3+5 (the 2–4h annotation task open since
2026-07-23). There is no model shortcut: **[MEASURED]** GPT-5.5-low scores F1 0.5125 on this
exact task, and worse, its errors are *systematically* the tag-text-vs-symbol bug — the variable
under test. A model-authored reference set would bake the bug into the control group and make
the A/B read null for the wrong reason.

### New strata and metric families

- Strata: `extent_source`, `tag_source`, node kind (symbol/port), on-sheet vs boundary edge.
- **New metrics, currently unmeasured:**
  - R0 extent-resolution IoU
  - **tag-association accuracy** — `tag_matching.py`'s nearest-OCR-word pairing has *no measured
    accuracy number today*. Needs one before we lean on it.
  - port-reference read accuracy
  - boundary-edge P/R/F1 (symbol↔port), **reported separately from on-sheet↔on-sheet, never
    averaged** — they're measured by different means (pure geometry vs geometry + text read),
    the same discipline as CLAUDE.md rule 5's two-part metric.

---

## 6. Deleted / retired

| Thing | Why |
|---|---|
| `_inflate_equipment_bboxes` + `EQUIPMENT_BBOX_PAD_INCHES` | Superseded by real extents; measured as partial-and-insufficient |
| Tier-1 #3's unseeded vector-geometry approach | Superseded by seeded resolution |
| Off-page-mentions-as-nodes | Replaced by `PortNode` |
| Tier-1 #4 as an active filter | Redundant by construction → defensive assert |

---

## 7. Phasing — each phase independently verifiable

- **Phase 0 — contract + plumbing, zero behavior change.** Land the dataclasses and dual-mode
  switch. **Gate: existing 12-sheet OPEN100 GT-injected numbers must reproduce EXACTLY.** Prove
  the refactor is neutral before changing any behavior.
- **Phase 1 — R0 + the decisive A/B.** Hand-verify extents for PX-2368's 3 clusters (~30–35
  entities; the 26-connection hand trace already exists as GT), rerun R1+R2a, rescore.
  **[PREDICTED] strict recall 58% → ~77%**, because all 5 current WRONG-ENDPOINT failures are
  textbook tag-text errors (LSHL-0100A/TSH-0100A bound to HAM instead of the vessel;
  PSV-0300A/FSV-0300C/PSV-0300C bound to XFMR/pumps instead of the treater). No GPU, no API
  spend, one variable changed, existing baseline.
- **Phase 2 — ports + boundary scoring + `agreement_diff` rewire.**
- **Phase 3 — R2a/R2b retune** (single-hop backbone, re-adjudicate backbone value, containment).
- **Phase 4 — re-benchmark:** 12-sheet OPEN100 `gt_injected` (regression) + 3 AG/RIVE `detected`.
- **Phase 5 — optional:** R4 retest on real extents; Opus/Fable as an extraction *arm*.

---

## 8. Open decisions — need Tom

1. **Port detection method** — deterministic border heuristic (reusing `diagram_area_bbox` /
   `bbox_center_outside`) vs training Molmo2 to point at connectors. §3.1.
2. **`PASSTHROUGH_TAG_TYPES`** `{valve, instrument, fitting, safety_device}` — frozen, still
   unreviewed (register gap #24).
3. **Do untagged symbols count as scoreable nodes** in `gt_injected` mode, or only as topology
   carriers?
4. **Backbone pass fate** — recommend keeping it and re-adjudicating after ports, not killing it
   now. Confirm.
5. **Full 492-candidate R4 run** — still open from register gap #23; my recommendation remains
   skip.
6. Register gaps #25 (backbone → `agreement_diff` AG/RIVE path) and #26 (Molmo2 + tag_matching
   wiring — *this proposal is the answer to #26*, so it needs sign-off).

## 9. Risks and unknowns — stated, not buried

- **Point→extent is not free.** Dense sheets pack symbols; a seed 20px off grows the wrong
  shape. Needs tolerance rules + the R0 IoU metric to catch it.
- **`tag_matching.py` accuracy is unmeasured.** The association problem shrinks from "which of
  5,608 shapes does this text name" (unsolvable) to "which nearby word belongs to this shape"
  (easy, not free) — but it can still mispair on dense sheets. **[MEASURED]** GD-B-540 had only
  **1** extractable embedded word (CAD outlined its text into paths), so association will fail
  outright on some sheets — which is exactly why untagged nodes must be kept.
- **Port detection unvalidated** (§3.1). Probe 3 validated reading, not finding.
- **R1's PID2Graph ceiling is unaffected** — corpus rendering mismatch, not an entity problem.
- **Dataset PID still can't run R1** (~7168×4561px, no downscale step; abandoned after 5+ min
  per sheet). Dense-sheet stratification stays untested.
- **No relation-KIND or hierarchy GT.** Unchanged. Still gated on human annotation hours. This
  proposal does **not** create ground truth — though it makes the annotation more valuable,
  since symbol boxes feed Molmo2 training *and* relation GT at once.
- **Adapter transfer unknown** for v3-stage13/v3-relation on symbol-extent crops.
- **Molmo2 retraining is not done.** Phase 1 deliberately does not wait on it — hand-verified
  extents de-risk the architecture bet independently.
