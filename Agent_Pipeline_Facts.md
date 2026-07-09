# P&ID Intelligence Agent — Code-Verified Pipeline Facts

**Source:** Claude Code session with direct access to
`rive-ai-platform/agents/pnid-intelligence-agent/` (the 15-stage knowledge-graph agent).
A separate `pnid-extraction-agent` exists with a different pipeline/output schema — none of
this applies to it. Obtained 2026-07-09 to unblock Stage4 checklist items 2.2/2.3/2.5.

All paths below are relative to `rive-ai-platform/agents/pnid-intelligence-agent/`.

---

## 1. Stage 3 — Tiling Scheme

Primary files: `pnid_agent/stages/tile_segmentation/grid.py`, `.../exclusion.py`,
`pnid_agent/models/tile_grid.py`, `pnid_agent/shared/coord_ops.py`

**Tile size & overlap**
- Tile size = **1024×1024 px**; overlap = **205 px**; stride = **819 px** (`tile_size - overlap`)
  - `grid.py:45-46` (defaults), `grid.py:71` (stride), `grid.py:9` (documented)
  - Config: `agent.yaml` → `stages.stage_03_tile_segmentation.tile_size_px: 1024`,
    `overlap_px: 205` — config and code default agree
- Overlap sized for symbols ≤ 200 px diameter (`grid.py:17-19`)

**Uniform vs adaptive**
- Grid is **uniform** — fixed 1024/205 over the resolved drawing area, same for every sheet
  (`grid.py:41-94`)
- What's adaptive: the drawing area being tiled (title block carved out, see below); edge tiles
  clamped smaller than 1024 and flagged `is_edge=True` (`grid.py:83-92`, `tile_grid.py:42-44`)
- Safety cap: `max_tiles_per_page: 200` (`agent.yaml`)
- **Nuance:** Stage 3 produces the full uniform grid, but Stage 4 doesn't necessarily run the
  LLM on every tile — a global overview pass selects a subset (`agent.yaml` →
  `stage_04...global_pass_enabled: true`). Replicating tile *inputs* = full uniform grid;
  replicating what the model actually *saw* requires accounting for overview-based selection.

**Title-block / border fencing**
- File: `exclusion.py`, function `resolve_drawing_area` (`exclusion.py:27-74`)
- Stage 2's title-block bbox subtracted from the page, padded by `margin_px = 20`
  (`exclusion.py:53-56`; config `title_block_margin_px: 20`)
- Sanity guard: if title-block bbox area > 30% of page area, rejected as unreliable and the
  whole page is tiled instead (`exclusion.py:60-64`; config `title_block_sanity_max_pct: 0.30`)
- Carving math: largest axis-aligned rectangle of (page − title-block-bbox), choosing among
  strip-above/below/left/right by area, ties prefer strip-above (`exclusion.py:77-99`)
- Three possible `DrawingAreaSource` labels per page: `title_block_excluded` /
  `title_block_rejected` / `full_page` (`tile_grid.py:22-26`)
- **Coordinate gotcha:** Stage 2's `TitleBlockRecord.bbox_drawing` is `[x, y, w, h]`, converted
  to `[x0, y0, x1, y1]` inside `resolve_drawing_area` (`exclusion.py:48`, `tile_grid.py:10-11`).
  Everything else in Stage 3 is `[x0, y0, x1, y1]`.

**Coordinate convention & tile→sheet remap**
- All Stage 3 bboxes are `[x0, y0, x1, y1]` in **original page-raster coords**
  (stage-00/pages/`p{i}.png` space) — `tile_grid.py:8-11,37-41`
- Tiles carry page-coord origins directly (grid computed in page space) — "drawing coords" ==
  "original page coords"; the title-block carve does not translate the origin
  (`coord_ops.py:13-17`)
- Remap function `coord_ops.tile_to_drawing` (`coord_ops.py:61-64`):
  ```python
  bbox_drawing = [x0+ox, y0+oy, x1+ox, y1+oy]  # then int(round(...))
  ```
  where `(ox, oy)` = the tile's top-left origin in page coords (`TileSpec.x0`, `.y0`). Inverse
  is `drawing_to_tile` (`coord_ops.py:67-70`). Rounding: `int(round(...))` (`coord_ops.py:36-37`)
- Applied at detection time: `detector.py:290`
  (`bbox_drawing = tile_to_drawing(bbox_tile, tile_origin=...)`)

---

## 2. Stage 4 — Symbol Detection Output Schema

Primary files: `pnid_agent/models/detections.py` (serialized artifact),
`pnid_agent/models/provenance.py`,
`pnid_agent/sub_agents/symbol_detection/ontology_render.py` (LLM tool schema), `.../driver.py`,
`.../nms.py`

Two distinct representations — **do not conflate them**:

### (a) What the LLM returns (tool-call input schema)

`ontology_render.py:87-183`, tool `detect_symbols`. Per symbol:

| Field | Type | Notes |
|---|---|---|
| `detection_id` | string | tile-local, e.g. `p1_t023_s00` |
| `entity_type` | string enum | enum = tenant ontology ids (§3); `ontology_render.py:110-124` |
| `bbox_tile` | int[4] | `[x0,y0,x1,y1]` **TILE-LOCAL**, min 0 (`ontology_render.py:125-131`) |
| `confidence` | number 0.0-1.0 | self-reported, uncalibrated (`ontology_render.py:132-137`) |
| `attributes`, `value`, `name`, `description`, `entity_subtype`, `library_hint_class_id` | — | `ontology_render.py:138-180` |

Required: `detection_id`, `entity_type`, `bbox_tile`, `confidence` (`ontology_render.py:182`).
Separately returns `associations[]` (symbol→OCR-word links by `span_id`) and
`unmapped_observations[]`.

### (b) The serialized artifact — `DetectionRecord` (treat this as ground truth for scoring)

`detections.py:116-203`, Pydantic model, `extra="forbid"`. Fields:

- `detection_id: str` — pattern `^p\d+_d\d{3,}$`, page-global, assigned in (y0,x0) order after
  NMS (`detections.py:121-124`)
- `page_index: int`
- `entity_type: str` — tenant ontology `semanticId` (`detections.py:126`)
- `entity_type_name: str?` — human-readable (`detections.py:127`)
- `source: str` — `"overview_pass"` or `"tile_pass"` (`detections.py:131`)
- `value: str?` (joined tag text), `name: str?`, `description: str?`, `entity_subtype: str?`
  (`detections.py:135-150`)
- `attributes: Dict[str,Any]` (`detections.py:151`)
- `provenance: AttributeProvenance` (`detections.py:158`) — **confidence and bbox live here**
- `raw_tile_id: str`, `raw_tile_bbox: int[4]` (tile-local, forensics only —
  `detections.py:166-171`)
- `library_hint_class_id: str?`, `rechecked: bool`, `recheck_reasoning: str?`
- `tag_type: str?`, `parsed_fields: Dict` (grammar-parse output — `detections.py:185-203`)

Top-level wrapper: `Stage04Output` at `stage-04/stage_04_output.json` (`detections.py:299-325`),
with `pages[].detections[]`.

### ⚠️ FLAG 1 — confidence is NOT top-level
`DetectionRecord` has no top-level `confidence`. It's `provenance.confidence` (float 0.0-1.0),
set at `driver.py:1159-1160`, model `provenance.py:73-77`. The internal `RawDetection`
(`nms.py:30`) *does* have top-level `.confidence` but is never serialized — it's mapped into
`provenance.confidence` when building `DetectionRecord`. **Read `detection["provenance"]["confidence"]`, not `detection["confidence"]`.**

### ⚠️ FLAG 2 — the symbol bbox is `provenance.bbox`, not top-level
No top-level `bbox` field. The sheet-coordinate box is `provenance.bbox` (`provenance.py:86`),
set at `driver.py:1162`. It's the projected symbol region in drawing/page coords, with an
**OCR-anchor correction**: if the LLM's box is disjoint from its associated tag words, it's
recentered on the OCR anchor (`driver.py:1142-1146`, `_anchor_correct_bbox`). So it's neither
the raw LLM box nor a pixel-union of OCR words. `raw_tile_bbox` (top-level) is tile-local
pre-projection, forensics only.

**Coordinate summary:** LLM emits tile-local `[x0,y0,x1,y1]` (`bbox_tile`). Serialized
`provenance.bbox` = absolute page/original coords, `[x0,y0,x1,y1]` (xyxy, not xywh, not
normalized). Projection via `tile_to_drawing` (§1). Confidence range 0.0-1.0, self-reported,
explicitly uncalibrated (`provenance.py:17-20,73-77`; `ontology_render.py:136`).

**Subtype/attribute fields beyond top-level type:**
- `entity_subtype` — drawing-convention subtype, e.g. "Gate Valve", "Reciprocating Pump"
  (`detections.py:147-150`)
- `attributes` — per-ontology-attribute values (`detections.py:151-157`)
- `tag_type` + `parsed_fields` — ISA grammar parse of the tag (`detections.py:185-203`)
- `library_hint_class_id` — ISA/ISO shape hint, non-authoritative (`detections.py:172-175`)

---

## 3. Entity Ontology

Primary files: `pnid_agent/sub_agents/symbol_detection/ontology_fetch.py`, `.../ontology_render.py`,
`pnid_agent/models/detections.py`

### ⚠️ FLAG 3 — the ontology is NOT a fixed/closed list in code
No hardcoded entity-type enum anywhere in the detection path. `entity_type` is a free string at
the model layer (`detections.py:22-27,126`). The set of valid types is **fetched per-tenant at
runtime** from the configurable-entities backend, parsed from each ontology entity model's
`semanticId`:
- Parse: `ontology_fetch.py:134-186`, `parse_ontology_response` →
  `entity_type = model.get("semanticId")` (`ontology_fetch.py:148`)
- Fetch entry points: `ontology_fetch.py:188` (`fetch_pid_ontology_with_payload`), `:211`
  (`fetch_pid_ontology`)

Runtime types injected two ways (`ontology_render.py`):
- Enum on the Anthropic tool schema — `build_detection_tool_schema` sets `entity_type.enum =
  sorted({schema.entity_type ...})` (`ontology_render.py:100,110-124`). This is where the
  "closed set" is enforced — but it's closed **per tenant, per run**, built from fetched data,
  not hardcoded.
- A prompt block listing each type's name/description/attributes (`ontology_render.py:36-84`)

Anything that doesn't fit a fetched type becomes an `UnmappedObservation`, never an entity
(`detections.py:206-220`, `ontology_render.py:96-98`).

**Hierarchy:** Flat at the `entity_type` level (tenant `semanticId` list). Specialization is
expressed via the separate `entity_subtype` free-text field (`detections.py:147-150`), not a
nested type tree.

**Illustrative examples** (not authoritative/exhaustive — code comments only):
`valve`, `safety_device`, `pipeline`, `measurement`, `nozzle`, `asset`
(`ontology_render.py:3-10`); tool-description examples: valve→`"valve"`, PSV→`"safety_device"`,
tank→`"asset"` (`agent.yaml` stage_04 comments).

**Fixed vs extensible:** Fully extensible / open per customer. Each tenant defines their own
types in configurable-entities; pipeline adapts at runtime, no code change. No default/fallback
hardcoded ontology — if a tenant defines none, the prompt says "(none — every detection becomes
an unmapped_observation)" (`ontology_render.py:43-47`).

---

## Summary of the three flags — implications for pid-ml

1. **Confidence** is at `provenance.confidence`, not top-level — a benchmark parser reading
   `detection["confidence"]` against the real artifact will miss it / KeyError.
2. **Symbol bbox** is at `provenance.bbox` (xyxy, absolute page coords, OCR-anchor-corrected),
   not a top-level `bbox`. `raw_tile_bbox` is tile-local forensics, not the answer.
3. **The entity ontology is runtime/per-tenant, not a closed enum.** Any pid-ml eval that
   assumes a fixed 32-class (or any fixed) type list is testing a different thing than what the
   agent actually does. The agent's typed-detection target is tenant-defined at run time — there
   is no single canonical ontology for Kaggle's 32 classes to be measured "coverage %" against.
   **This needs a benchmark-design decision, not a code fix** — see `Stage4_Checklist_Status.md`
   item 2.2.

All three are cases where prose documentation (and the internal `RawDetection` shape) present a
simpler/flatter structure than the actual serialized `DetectionRecord` / runtime ontology.
