# Conversion Layer — Build Plan

**Date:** 2026-07-16
**Status: BUILT AND VERIFIED (2026-07-16).** This plan was executed the same day it was
written. The full conversion layer lives at `src/e2e_bench/`, all 25 tests pass
(`src/e2e_bench/tests/`, including a full smoke chain through 8 real stages), and every
converter has been run against the REAL agent code (not mocked) on a real P&ID image. See
`src/e2e_bench/PHASE0_REPORT.md` for the environment/import findings and
**"Build Log — real discoveries" at the bottom of this file** for everything that surfaced
only by actually running the code (several real bugs and undocumented agent requirements
that would have silently broken a harness built on assumptions alone). The rest of this
document is left as originally written (the plan), since it turned out accurate; read the
build log at the end for what changed in practice.

**Purpose:** the glue that maps local model outputs (Qwen3-VL, Molmo2, PaddleOCR) into the
real `pnid-intelligence-agent`'s exact internal schemas, so the agent's own deterministic
code (tiling, line tracing, graph construction, skid merge) runs **unmodified** in the
two-arm end-to-end benchmark (Arm P = GPT-5.5-low, Arm L = local models). This plan is
written to be executed by a separate session — every contract below was extracted from the
actual agent source with file:line citations on 2026-07-16; **verify each citation still
matches before building on it** (the agent repo may have moved since).

**Agent repo root (all paths below relative to it unless stated):**
`/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-intelligence-agent`

**This repo (where the conversion layer lives):** `/Users/tomgeorge/pid-ml`, new package
`src/e2e_bench/`.

**Compute:** everything in this plan is CPU-only and runs on the Mac. No Colab, no GPU.
(Model *inference* happens elsewhere; converters only reshape already-produced outputs.)

---

## 0. The load-bearing discovery

The agent's artifact layer has a pure-filesystem implementation:
`LocalFsArtifactStore(root=...)` (`storage/local_fs.py:36-45`, or env `PNID_RUNS_DIR`).
Artifacts live at `{root}/{job_id}/stage-XX/...`. **No orchestrator, no Keycloak, no
platform services are needed** to feed the agent's deterministic stages — the harness
constructs a `DrawingDocument`, writes correctly-shaped stage artifacts via the conversion
layer, and calls the agent's own functions.

`DrawingDocument` (`models/drawing_document.py:110-118`): `doc_id`, `source`,
`pages: List[DrawingPage]` (each: `page_index`, `raster`, `normalization`,
`page_classification`), `tenant_id`, `job_id`.

**Environment:** Python ≥3.10, pydantic ≥2.0 (`pyproject.toml:5,12`). Most models are
`ConfigDict(extra="forbid")` — an extra key is a hard error. Emit exactly the fields listed.

---

## 1. Decision log (settled — do not re-litigate, but flag if code contradicts)

| # | Decision | Rationale |
|---|---|---|
| D1 | Import agent code in place via path dependency (`sys.path` / editable install of the agent package + monorepo `shared/entity_operations` and `rive_adk`), never vendor/copy | Same machine, keeps us pinned to real code; vendoring drifts |
| D2 | Fixed benchmark ontology, authored as a JSON file in pid-ml, loaded via `OntologyRelationIndex.from_entries()` (`stages/graph_construction/ontology_relation_index.py:115`) | The real ontology is per-tenant at runtime (Agent_Pipeline_Facts.md flag 3); a benchmark needs a frozen, documented one. MUST cover every (source_type, target_type) pair local models can emit — unmatched pairs silently emit NO relation (`unresolved.missing_relation_schema`) |
| D3 | Molmo2 emits points, no confidence → pseudo-box ±20px around point (matches `src/stage4_symbol_detection/molmo_candidate.py` in pid-ml), confidence fixed 0.5 | `RawDetection.confidence` is required float; assembly drops confidence==0.0 records (`driver.py:1047-1177`), so never emit 0.0 |
| D4 | Title-block substitute may return `located=False, bbox_drawing=None` | Stage 3 safely falls back to full-page tiling on None/invalid bbox (`stages/tile_segmentation/exclusion.py:28-75`) — legitimate agent path, not a hack |
| D5 | Title-block `fields` keys use our 4 benchmark names (`drawing_number`, `revision`, `title`, `site`) declared as the benchmark "tenant schema" | Real keys are tenant-runtime (`extraction_schema.attribute_names`, `title_block_extraction/driver.py:304`) — no fixed list exists in code; we declare ours and document it |
| D6 | Reuse agent NMS (`nms.py`) — import is clean. For `_compose_detection_records`, TRY importing from `driver.py` first; if the import chain fails in our env (it pulls detector→anthropic at import time, though no client is constructed), replicate the ~130-line pure function into `e2e_bench/vendor_compose.py` with a header comment citing `driver.py:1047-1177` and a test comparing against the original if importable | Prefer real code; have a documented fallback |
| D7 | Local relation-stage substitutes write verdicts; the harness reuses the agent's stage-12/13 WRITE semantics (copy-then-filter) rather than inventing its own merge | Keeps arm outputs byte-comparable to real agent artifacts |
| D8 | Skid substitute writes the sidecar JSON directly (`stage-10.5/skid_groups.json`); do NOT try to run `sub_agents/skid_grouping/driver.py` (needs `vlm_runner` + rive_adk) | The sidecar is the interface; `infer_from_skid_groups` (`graph_construction/inference.py:705-726`) consumes it cleanly |
| D9 | Both arms (GPT-5.5 and local) go through the SAME converters — converters take a plain "normalized model answer" dataclass, not raw model text. Backend-specific parsing (Molmo `<points>` regex, Qwen JSON, GPT JSON) lives in `backends/`, separate from converters | One conversion path = no arm-specific schema bugs |
| D10 | Every parse failure is counted and surfaced (per stage, per sheet), and falls back to the documented degenerate value (empty detections, `located=False`, all-`uncertain` verdicts). Never a silent drop, never a fabricated record | The skid-matrix run proved fallbacks can masquerade as real scores; the harness must report `parse_failures` alongside every number |

---

## 2. Package layout

```
pid-ml/src/e2e_bench/
  __init__.py
  agent_env.py          # locates agent repo, sys.path setup, import checks (Phase 0)
  benchmark_ontology.json   # D2: frozen entity types + relation entries
  ontology.py           # loads benchmark_ontology.json -> OntologyRelationIndex + type-name lookup
  types.py              # normalized answer dataclasses (backend-agnostic, D9)
  backends/
    __init__.py
    parse_molmo.py      # <points coords=...> -> List[Point]  (port from pid-ml src/stage4_symbol_detection/molmo_candidate.py::parse)
    parse_qwen_json.py  # fenced/loose JSON extraction (title block dict, per-symbol skid dict, keep/remove, yes/no)
    parse_gpt_json.py   # same shapes for Arm P (usually shares code with parse_qwen_json)
    parse_paddle.py     # PaddleOCR result object -> List[NormalizedWord]
  converters/
    __init__.py
    stage01_classification.py
    stage015_ocr.py
    stage02_titleblock.py
    stage04_detection.py
    stage105_skid.py
    stage13_entity_validation.py
    stage12_relation_validation.py
  assembly/
    __init__.py
    document.py         # build DrawingDocument + LocalFsArtifactStore run dir
    entities.py         # DetectionRecord -> EntityMapping -> build_entity -> BundleEntity
    compose_fallback.py # D6 fallback copy of _compose_detection_records (only if needed)
  tests/
    fixtures/           # hand-authored raw model outputs + expected artifacts
    test_stage01.py ... test_stage12.py
    test_roundtrip.py   # every artifact re-validates through the agent's own pydantic models
    test_smoke_chain.py # one sheet end-to-end through deterministic stages with mock model answers
```

---

## 3. Phase 0 — environment + import verification (do this first, fail fast)

1. Create `agent_env.py`: resolves the agent repo path (env var
   `PNID_AGENT_REPO` with the default above), inserts into `sys.path`, and exposes
   `import_agent(module_path)`.
2. Verify these imports succeed in pid-ml's venv and record results in the build log:
   - `models.page_classification`, `models.page_ocr`, `models.title_block`,
     `models.detections`, `models.line_tracing`, `models.rive_ontology`,
     `models.ontology_mapping`, `models.relation_validation`,
     `models.drawing_document`, `models.provenance` — pydantic only, should be clean
   - `shared.coord_ops`, `stages.tile_segmentation.grid`, `stages.tile_segmentation.exclusion` — pure
   - `sub_agents.symbol_detection.nms` — imports only coord_ops
   - `stages.line_tracing.driver` — needs numpy/cv2/scikit-image/PIL installed in venv
   - `stages.graph_construction.relations`, `.ontology_relation_index`, `.inference` — clean
   - `stages.graph_construction.entities` — needs `entity_operations`
     (monorepo `rive-ai-platform/shared/entity_operations`, pip-installable locally)
   - `sub_agents.symbol_detection.driver` (for `_compose_detection_records`) — pulls
     anthropic-SDK chain at import; **may fail** → triggers D6 fallback
   - `stages.graph_construction.driver` (`stage_11_run`), `sub_agents.entity_validation.driver`,
     `sub_agents.relation_validation.driver` — need `rive_adk` importable
     (via `shared/logging_compat`, `logging_compat.py:15`); no network at import
3. `pip install -e` (or path-install) `entity_operations` and `rive_adk` from the monorepo
   into pid-ml's venv. Record exact versions installed (no lockfile exists in the agent dir —
   only `>=1.0.0` bounds; pin whatever resolves, in a `constraints-e2e.txt`).
4. Deliverable: `agent_env.py` + a `PHASE0_REPORT.md` listing each import PASS/FAIL and the
   chosen path for D6.

**Stop condition:** if `rive_adk` or `entity_operations` cannot be installed/imported, stop
and report — stages 11/13/12 full drivers are blocked (converters for 01→06 can still
proceed; `build_relations` + `infer_from_skid_groups` are importable without them).

---

## 4. Phase 1 — normalized answer types + backend parsers

`types.py` dataclasses (all coordinates in ORIGINAL PAGE-RASTER pixels, xyxy ints unless
noted — matching `Agent_Pipeline_Facts.md` §1's convention):

```python
@dataclass class NormalizedWord:        # from PaddleOCR / Google Vision alike
    text: str; bbox: list[int]          # xyxy ints, page coords
    confidence: float
@dataclass class NormalizedDetection:   # from Molmo2 points or GPT boxes, TILE-LOCAL
    bbox_tile: list[int]                # xyxy ints, tile-local (Molmo: point ±20 pseudo-box, D3)
    confidence: float                   # Molmo: 0.5 fixed (D3); never 0.0
    entity_type: str                    # must be a benchmark-ontology semanticId (D2)
    value: str | None = None
@dataclass class NormalizedTitleBlock:
    located: bool
    fields: dict[str, str | None]       # keys: D5 benchmark names
@dataclass class NormalizedSkidAssignment:
    asset_temp_id: str
    members: list[tuple[str, str | None, float]]  # (target_temp_id, forward_relation_name|None, confidence)
@dataclass class NormalizedEntityVerdict:
    temp_id: str; keep: bool; confidence: float
@dataclass class NormalizedRelationVerdict:
    relation_id: str; verdict: str      # confirmed|rejected|uncertain
    revised_confidence: float
@dataclass class ParseOutcome[T]:       # wrapper every backend parser returns
    value: T | None; parse_failed: bool; raw_text: str; error: str | None
```

Backend parsers (`backends/`) each return `ParseOutcome`. Port the Molmo `<points>` parser
from pid-ml `src/stage4_symbol_detection/molmo_candidate.py::parse` (it has the
duplicated-leading-index repair; keep it). JSON extraction: fenced-block first, then first
balanced `{...}`/`[...]`; on failure return `parse_failed=True` — the converter applies the
D10 degenerate fallback and the harness counts it.

---

## 5. Phase 2 — the seven converters (exact contracts)

Every converter: `(normalized answers, context) -> agent artifact(s) written via
LocalFsArtifactStore` and the same pydantic object returned for chaining. Every converter
validates its own output by round-tripping through the agent's model class before writing
(pydantic `extra="forbid"` makes this the correctness gate).

### 5.1 `stage01_classification.py`
Target: `stage-01/stage_01_output.json` = `Stage01Output` (`models/page_classification.py:43-55`).
- Required fields: `doc_id`, `model_version`, `total_duration_ms`, `total_cost_usd`,
  `applied_treat_unknown_as` (enum `pid_drawing/legend/cover/notes/index/pfd/other`),
  `confidence_threshold_for_skip`, `classifications[]`, `pages_to_process`, `pages_skipped`.
- `PageClassificationRecord` (`:25-40`): `page_index`, `classification`, `confidence`,
  `model_version`, `duration_ms`, `cost_usd` required.
- Downstream consumes ONLY `pages_to_process` (+ cost/duration/model_version bookkeeping)
  — `agent.py:498-504`. `pages_to_process` = pages classified `pid_drawing` or `legend`.
- **CRITICAL:** also set `page.page_classification` on the in-memory `DrawingDocument`
  (later stages gate on `DrawingPage.page_classification == PID_DRAWING`) — writing the
  JSON alone is not enough.
- Benchmark simplification: our sheets are known P&IDs; the substitute model's answer maps
  to the enum; cost/duration filled from real call telemetry (0 allowed).

### 5.2 `stage015_ocr.py`
Targets: `stage-01.5/stage_01_5_output.json` = `Stage015Output` (`models/page_ocr.py:90-104`)
AND per-page `stage-01.5/intermediate/p{i}_words.json`.
- Per-page file = JSON **list** of `OcrWord.to_dict()` dicts (`page_ocr.py:41-53`):
  `{"text": str, "bbox": [x0,y0,x1,y1] ints page coords, "confidence": float}`.
  `OcrWord` is a plain dataclass (not pydantic) — `page_ocr.py:19-39`.
- `PageOcrRecord` (`:67-87`): `words_uri` MUST be `stage-01.5/intermediate/p{i}_words.json`
  (`stages/ocr_page/driver.py:126,140`); fill `n_words`, `per_page_word_counts`.
- **Word order is load-bearing:** `AttributeProvenance.source_word_indices` and stage-4
  association `span_id`s (`p{page}_w{global_idx}`, `detector.py:491`, `tile_words.py:25-36`)
  index into this list. Freeze the order at write time; never re-sort afterward.
- PaddleOCR mapping: its line/word results → word-level entries; polygonal boxes → axis-aligned
  int xyxy envelope; keep Paddle's recognition confidence.

### 5.3 `stage02_titleblock.py`
Target: `stage-02/title_block_output.json` = `Stage02Output` (`models/title_block.py:85-102`).
- `TitleBlockRecord` (`:59-82`): `page_index`, `located` required; `bbox_drawing` is
  **`[x, y, w, h]` xywh** (NOT xyxy — `:66-69`); `fields: Dict[str, TitleBlockField]`.
- `TitleBlockField` (`:44-56`): `value`, `grammar_violation=False`,
  `provenance: AttributeProvenance` REQUIRED (its required members: `confidence` 0-1,
  `source` str — `models/provenance.py:63-109`; use `source="e2e_bench_local"` /
  `"e2e_bench_gpt55"`).
- Our substitutes don't localize the block → D4: `located=False`, `bbox_drawing=None`,
  but still emit `fields` when the model extracted values (verify stage 3 only reads
  bbox, not `located`, at `exclusion.py:28-75` — cite in code comment).

### 5.4 `stage04_detection.py` (the big one)
Pipeline per page: tile with the agent's OWN grid
(`stages/tile_segmentation/grid.py`, 1024/205 — or the Molmo 512/102/upscale2/enhance
config for Arm L, recorded in run metadata) → run model per tile (outside this layer) →
`NormalizedDetection`s per tile → this converter:
1. Build `RawDetection` per answer (`sub_agents/symbol_detection/nms.py:29-52`):
   `entity_type`, `entity_type_name` (from benchmark ontology lookup), `confidence`,
   `bbox_drawing` = `tile_to_drawing(bbox_tile, tile_origin)` (`shared/coord_ops.py:61-64`;
   for the upscaled Molmo config divide by upscale factor BEFORE the tile-origin offset —
   same math as the Stage4 detection notebook), `value`, `attributes={}`,
   `source_word_indices=[]` or matched word indices, `raw_tile_id` (`p{p}_t{NNN}` format),
   `raw_tile_bbox` (tile-local), `library_hint_class_id=None`.
2. `dedup_across_tiles(...)` (`nms.py:77-159`) with its defaults (IoU 0.5) → deterministic
   `p{page}_d{NNN}` ids in (y0,x0) order. Associations: skip in v1 (empty list) unless
   time allows OCR-word matching; then `relink_associations` (`nms.py:162-189`).
3. Assemble `DetectionRecord`s: import `_compose_detection_records`
   (`sub_agents/symbol_detection/driver.py:1047-1177`) per D6, passing `page_words` from
   stage-01.5 output (enables the agent's own OCR-anchor bbox correction) →
   `Stage04Output` wrapper (`models/detections.py:299-325`) → write
   `stage-04/stage_04_output.json`.
- Remember flags 1-2: confidence/bbox land under `provenance.*` — the assembly function
  handles this; do NOT construct DetectionRecord by hand.

### 5.5 `stage105_skid.py`
Target: `stage-10.5/skid_groups.json` (`sub_agents/skid_grouping/driver.py:9-26,195-204`):
```json
{"schema_version": "1.0", "doc_id": "...", "pages_processed": [0],
 "groups": {"<asset_temp_id>": [{"target_temp_id": "p0_e0023",
     "forward_relation_name": "<name-or-null>", "confidence": 0.92, "reasoning": "..."}]},
 "telemetry": {}}
```
- Input to the substitute model (harness side, mirroring the real driver
  `driver.py:51-70`): per `asset`-type entity — ROI crop (padding 0.6×, min 250px, max long
  side 1200px), candidate entities inside the ROI, and the benchmark ontology's relation
  options. Reuse the per-symbol prompt format that scored 92.3%.
- Merge path: `infer_from_skid_groups(entities, ontology_relations, groups, *,
  page_index, starting_relation_position)` (`stages/graph_construction/inference.py:705-726`).
- `forward_relation_name` must be one of the benchmark ontology's names or null.

### 5.6 `stage13_entity_validation.py`
- Reads `stage-11/rive_ontology.json` (`RiveOntology`, `models/rive_ontology.py:470-505`).
- Substitute (v3-stage13 adapter / GPT) emits keep/remove per entity →
  `NormalizedEntityVerdict`s → map to the real write semantics
  (`sub_agents/entity_validation/driver.py:306-333,483-488`):
  removed = delete from `entities`; their relations KEPT but stamped
  `review_status="rejected"` + `metadata.validation_status="rejected"`; write full corrected
  copies to `stage-13/rive_ontology.json` + `stage-13/drawing_document.json`, then
  `stage_13_output.json` last (atomicity marker).
- Two implementation options, in preference order: (a) call the real `stage_13_run` with an
  injected `vlm_runner`-equivalent if its signature permits (check driver), or (b) replicate
  ONLY the write/merge logic (cited lines) in the converter. Investigate (a) first; the
  extraction confirmed the LLM emission contract is the `validate_entities` tool schema
  (`entity_validation/tool_schema.py:17-124`) — mapping keep/remove →
  `removed_temp_ids` + confidence-0 reclassifications reproduces the exact path.

### 5.7 `stage12_relation_validation.py`
- Reads stage-13's `rive_ontology.json` (parameter `source_rive_uri`); only relations with
  `confidence < threshold` (0.75 per the run signature `relation_validation/driver.py:57-75`
  — the docstring's 0.9 is wrong, trust the signature) get validated.
- Per relation: `NormalizedRelationVerdict` → `RelationValidation`
  (`models/relation_validation.py:19-40`).
- Write semantics: confirmed → confidence bumped; rejected → removed from `relations` +
  logged in `suggestions.unresolved` (kind=`llm_rejected_relation`); uncertain → kept with
  revised confidence. Write `stage-12/rive_ontology.json`, then `stage-12/stage_12_output.json`
  (`Stage12Output`, `relation_validation.py:43-80`) LAST.
- Same (a)/(b) choice as 5.6.

### Also needed: entity assembly between stage 4 and the graph stages
`assembly/entities.py`: `DetectionRecord` → `EntityMapping`
(`models/ontology_mapping.py:45-72`: `mapping_id ^p\d+_m\d{3,}$`, `source_detection_id`,
`page_index`, `entity_type`, `entity_type_name`, `attribute_schema=[]`, `match_confidence`,
`match_method="passthrough_v1"`) → `build_entity(mapping, detection, *, temp_id, page_size,
...)` (`stages/graph_construction/entities.py:183-204`) → `BundleEntity`
(`rive_ontology.py:237-338`; **entities with no bbox are dropped** — assert every detection
has `provenance.bbox` before this step). Then `build_relations(page_graph,
detection_to_temp_id, entity_type_by_temp_id, ontology_relations, ...)`
(`stages/graph_construction/relations.py:68-85`) → `BundleRelation`s
(`rive_ontology.py:340-421`, camelCase aliases, `populate_by_name=True`).
Prefer calling the full `stage_11_run` (`graph_construction/driver.py:87-127`) with
`ontology_payload_factory` injected (D2) and `token_provider=None` (direct fetch skipped,
`driver.py:1817-1820`) — falls back to hand-chaining the pieces above if the driver's
config surface is too heavy.

### Stage 6 (no converter — sequencing constraint)
`stage_06_run` (`stages/line_tracing/driver.py`) is deterministic and importable, but it
**reads `stage-04/stage_04_output.json`** for symbol masking (`driver.py:167-184`).
Sequencing: stage-04 artifact must be written before stage 6 runs. No LLM, no conversion.

---

## 6. Phase 3 — benchmark ontology file (D2)

Author `benchmark_ontology.json`:
- Entity types: the set local models actually emit. Baseline:
  `valve`, `instrumentation`, `pump`, `tank`, `general`, `inlet/outlet` (PID2Graph labels,
  confirmed in that dataset's graphml) + `asset` (required by the skid driver's
  asset-type filter). Each: `semanticId`, human name.
- Relation entries per `OntologyRelation` (`ontology_relation_index.py:30-52`):
  `source_entity_type`, `target_entity_type`, `forward_relation_name`,
  `reverse_relation_name`, `cardinality` (use `MANY_TO_MANY` default), optional
  `line_type_hint`. **Cover ALL ordered type pairs** with a generic
  `connects to`/`connected from` — unmatched pairs are silently dropped (§1 D2). Plus one
  `Installed Valves`-style relation for skid membership naming.
- Document in the file header that this is the benchmark's frozen stand-in for the
  per-tenant runtime ontology (Agent_Pipeline_Facts.md flag 3).

---

## 7. Phase 4 — validation + tests

1. **Round-trip test (the core gate):** every converter output must
   `Model.model_validate(json.loads(written_bytes))` through the agent's OWN pydantic
   class with zero errors (`extra="forbid"` catches shape drift). Applies to:
   Stage01Output, Stage015Output, Stage02Output, Stage04Output, Stage06Output (consumed),
   RiveOntology, Stage12Output.
2. **Fixture tests per backend parser:** hand-authored raw model outputs (take real ones
   from this session's runs where available — e.g. real Molmo `<points>` strings from the
   detection notebook, real Qwen JSON from the skid matrix run on HF
   `benchmarks/skid_matrix_molmo2_qwen_adapters_v1.json` `predictions` key) + malformed
   variants (truncated JSON, missing keys, out-of-range indices) asserting
   `parse_failed=True` and the D10 fallback.
3. **Coordinate tests:** tile→page remap identity checks (a synthetic detection at tile
   corner maps to the expected page pixel, upscale config divides correctly), xywh vs xyxy
   for `bbox_drawing` in TitleBlockRecord vs everywhere else.
4. **ID-format tests:** regex conformance for every generated id
   (`p\d+_d\d{3,}`, `p\d+_m\d{3,}`, `p\d+_r\d{4}`, span_id `p\d+_w\d+`,
   segment `p\d+_g\d{4,}` consumed).
5. **Smoke chain test (`test_smoke_chain.py`):** one PID2Graph `Complete` sheet, MOCK
   normalized answers (no model calls): stage01 → stage015 (empty words OK) → stage02
   (`located=False`) → stage04 (3 hand-placed detections) → run REAL `stage_06_run` →
   entity assembly → REAL `build_relations` with the benchmark ontology → stage105 sidecar
   (one group) → REAL `infer_from_skid_groups` → stage13 converter (remove 1 entity) →
   stage12 converter (reject 1 relation). Assert: final `stage-12/rive_ontology.json`
   validates, has expected entity/relation counts, and the run dir layout matches the real
   agent's (`{root}/{job_id}/stage-XX/...`).
6. **If D6 fallback was used:** a conditional test comparing `compose_fallback` output
   against the real `_compose_detection_records` (skipped when the import fails).

---

## 8. Explicit non-goals (v1)

- No OCR-word↔detection association matching (empty `associations` — costs the agent's
  anchor-correction some accuracy; note it in run metadata as a known simplification and
  revisit only if detection-stage attribution flags it).
- No overview-pass / tile-subset selection (Agent_Pipeline_Facts.md §1 nuance) — full grid.
- No stage-05 OCR ensemble (off in prod), no dynamic-cropping stage (off by default).
- No attempt to reproduce cost/latency accounting beyond filling required numeric fields.
- The harness that CALLS models (two arms, cascade/isolated modes, scoring) is a separate
  plan — this plan covers only the conversion layer those arms share.

---

## 9. Open items the executor must resolve (and report back, not guess)

1. Whether `sub_agents/symbol_detection/driver.py` imports cleanly in pid-ml's venv (D6).
2. Whether `stage_13_run`/`stage_12_run` accept an injectable runner cleanly enough for
   option (a) in §5.6/5.7 — read their signatures before choosing (a) or (b).
3. Exact `rive_adk`/`entity_operations` versions that resolve (§3.3) — record in
   `constraints-e2e.txt`.
4. `Stage06Output` is listed as consumed-only here; confirm `stage_06_run` writes it itself
   (expected) so no converter is needed.
5. Whether `stage_11_run`'s config surface is practical (§5 assembly) or hand-chaining is
   cleaner — try the driver first, time-box it.

## 10. Suggested execution order

Phase 0 (env) → 6 (ontology file) → 1 (types+parsers) → 5.1/5.2/5.3 (simple converters)
→ 5.4 (detection) → assembly + stage-6 sequencing → 5.5 → 5.6/5.7 → Phase 4 tests
throughout (round-trip test lands with each converter, smoke chain last).

---

## 11. Build log — real discoveries (2026-07-16)

Everything below was found by actually running the code, not by reading it. Kept here
verbatim as the authoritative record for anyone building the harness on top of this layer.

**Environment.** System Python was 3.9.6; the agent needs ≥3.10. Used pyenv's 3.12.10.
`rive_adk` depends on a PyPI-nonexistent `rive-security>=0.1.0` — satisfied by editable-
installing the monorepo's `shared/security` package (which declares itself as that PyPI
name). Full working install order in `PHASE0_REPORT.md`.

**Every import passed, including the two flagged as risky.** D6's fallback (hand-replicating
`_compose_detection_records`) was never needed — the real function imports and runs
cleanly. `stage_11_run`/`stage_13_run`/`stage_12_run` (needing `rive_adk`) also imported and
ran cleanly. One harmless logged (not raised) error appears on every `rive_adk`-touching
import/call: `rive_adk.core.config` complains about a missing `AgentConfig.agent` field.
Never blocked anything observed; still unresolved as to whether it's truly inert.

**Option (a) for stage 13/12 confirmed, and simpler than the plan guessed.** `vlm_runner`
only needs `_get_messages_client()` returning an object with an async
`.messages.create(**kwargs)`. Both stages' response parsing is defensive
(`getattr(x, "attr", None)` with dict `.get()` fallback), so the fake response can be a
`SimpleNamespace`/plain dicts — no real `anthropic` SDK types needed. Built as
`assembly/fake_llm.py` (`FakeMessagesClient`/`FakeRunner`), ~60 lines, shared by both
converters. D6-style hand-reimplementation of the write/merge logic was never needed either.

**Stage 13 vs stage 12 have different call shapes — this matters for correlation.** Stage 13
makes ONE call per PAGE covering every entity in one JSON payload
(`reclassifications`/`attribute_corrections`/`confirmed_ok`/`removed_temp_ids`) — no
per-call correlation needed, the fake client just always returns the same aggregated
payload. Stage 12 dispatches CONCURRENTLY via `asyncio.gather`, one call per relation, and
**the prompt text contains no `relation_id`** — only `source_label`/`target_label`/
`relation_name`/`pipeline_confidence` (confirmed by reading
`relation_validation/prompt.py::build_user_message`). The converter correlates on that
4-tuple. **Known real limitation:** two relations sharing an identical tuple (e.g. two
unlabeled entities with the same relation name and confidence) are ambiguous to this fake
client — not fixable without touching agent code (out of scope, D1). Rare but worth
watching if stage-12 results ever look wrong specifically on sheets with many unlabeled
entities.

**`stage_13_run` needs `context.get_fresh_token()` unless you use `schema_factory`.**
Without it, `_resolve_schemas` (`stages/ontology_validation/driver.py:222-262`) raises
`Stage10Error("no bearer token available")`. `schema_factory=lambda: []` is the documented
"test path" bypass (checked first, before any token/backend logic) and works fine for a
benchmark that only does keep/remove (no attribute-schema-driven corrections).

**`context` is genuinely permissive everywhere observed.** Every real usage found across
stage 11/13/12 was `getattr(context, "attr", None)` with a default — a bare
`SimpleNamespace(tenant_id="benchmark")` satisfies all of them. No richer context object
needed for the benchmark.

**`Raster.uri` must be relative to the job's artifact directory, not an absolute path.**
`LocalFsArtifactStore._abs()` raises `ValueError("... escapes job_dir ...")` on any path
resolving outside `{root}/{job_id}/`. Running `stage_06_run` with a `DrawingDocument` built
from an absolute image path failed with `RasterDecodeFailed`. Fixed by having
`assembly/document.py` COPY the source image into the store at the real convention path
(`stage-00/pages/p{i}.png`, matching `Agent_Pipeline_Facts.md` §1) and setting `Raster.uri`
to that relative path instead.

**Multiple real stages gate on `DrawingPage.page_classification == PID_DRAWING`, not just
the ones documented in §5.1.** `stage_06_run` silently skips a page (produces an EMPTY
`pages` list, no error) if classification wasn't set — confirmed by grep
(`line_tracing/driver.py:201-202`). A smoke test that skips stage 1 "for speed" will look
like it passed (no exception) while silently processing zero pages downstream. Always run
stage 1's converter first, even in isolated per-stage testing.

**The single most important discovery, load-bearing for the harness (not just this layer):
a detection needs a non-empty `value` (tag/label text) or its entity is silently dropped.**
`build_entity` (`stages/graph_construction/entities.py`) requires a derived `name` to
construct a `BundleEntity` at all: `derived_name = detection.name or
_derive_clean_label(detection)`, and `_derive_clean_label` needs a grammar-reconstructed
tag, a Tag ID attribute, or a single-token raw `value`. With none of those, `build_entity`
returns `(None, None, suggested)` — no error, no entry in `unresolved_out`, just silently
gone before ever reaching stage 6/11/13/12. **Confirmed by direct test:** a detection with
`value=None` produced zero entities; the identical detection with `value="V-101"` produced
one. Documented prominently in `types.py`'s `NormalizedDetection` docstring because of its
consequence for the harness: **Molmo2's native output (points only, no text) will produce
ZERO usable entities on its own.** Either pair Molmo2 with a real OCR-tag-matching step
(the v1 non-goal, §8 — worth reconsidering given this), or accept that a Molmo2-only arm
tests detection localization only, never contributes anything to the entity/relation/skid
stages of an end-to-end score. This is a real design decision for whoever builds the
harness next, not an implementation detail.

**All of the above is now covered by regression tests** — `tests/test_backend_parsers.py`
(17 tests, including the JSON-bracket-order bug below), `tests/test_converters_roundtrip.py`
(7 tests), `tests/test_smoke_chain.py` (1 test, the full 8-stage chain). 25/25 pass.

**One real bug caught and fixed during the build (not agent-side — ours):** the JSON
extractor's loose-fallback path tried `{...}` before `[...]` unconditionally, so a top-level
JSON array of objects (the skid-assignment shape) got truncated down to just its first
nested object. Caught by the parser's own smoke test before it ever reached a converter.
Fixed in `backends/parse_json_common.py::_extract_json_text` (now picks whichever bracket
appears first in the text) and locked in as `test_skid_json_top_level_array_of_objects`.

---

## 12. Scoping note — what comes after this (explicitly NOT built yet)

This plan/build covers the **conversion layer only**. The harness that actually calls
models (Arm P / Arm L, cascade vs. isolated-per-stage modes, scoring against a file-disjoint
PID2Graph holdout, per-stage error attribution) is a deliberate, separate follow-up — it
builds ON TOP of `src/e2e_bench/converters` and `src/e2e_bench/assembly`, using them as a
library. Do not conflate the two: this conversation ends at "the glue works," not at "the
benchmark runs." The value-required-for-entities finding above and the stage-12 correlation
limitation are the two things that most directly shape that next plan's design and should
be read before writing it.
