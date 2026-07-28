"""
Backend-agnostic normalized answer types (Conversion_Layer_Plan.md D9, section 4).

Every backend parser (backends/parse_*.py) returns model answers wrapped in these types,
regardless of whether the underlying call was Qwen, Molmo2, PaddleOCR, or GPT-5.5. The
converters (converters/*.py) consume ONLY these types, never raw model text — this is what
keeps Arm P (GPT-5.5) and Arm L (local) sharing one conversion path (D9), so there is no
per-arm schema drift.

All coordinates are ints in the ORIGINAL PAGE-RASTER pixel space unless a field name says
otherwise (e.g. `bbox_tile` is tile-local) — matching Agent_Pipeline_Facts.md's convention.
All bboxes are xyxy [x0, y0, x1, y1] EXCEPT NormalizedTitleBlock, which is written to the
agent's TitleBlockRecord.bbox_drawing field and that field is genuinely xywh
(models/title_block.py:66-69) — kept as xyxy here at the normalized-type level for
consistency, converted to xywh only inside the stage02 converter, right at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class ParseOutcome(Generic[T]):
    """What every backend parser returns. `parse_failed=True` means the converter must
    apply its documented degenerate fallback (D10) and the harness must count this failure
    — never silently treat a fallback value as a real result (see the skid-matrix lesson:
    two adapters scored 79.1% that was 100% parse failure in disguise)."""
    value: Optional[T]
    parse_failed: bool
    raw_text: str
    error: Optional[str] = None

    @classmethod
    def ok(cls, value: T, raw_text: str) -> "ParseOutcome[T]":
        return cls(value=value, parse_failed=False, raw_text=raw_text, error=None)

    @classmethod
    def failed(cls, raw_text: str, error: str) -> "ParseOutcome[T]":
        return cls(value=None, parse_failed=True, raw_text=raw_text, error=error)


@dataclass
class NormalizedWord:
    """One OCR word. -> converters/stage015_ocr.py -> OcrWord.to_dict() shape
    (models/page_ocr.py:19-53)."""
    text: str
    bbox: list  # [x0, y0, x1, y1] ints, page coords
    confidence: float


@dataclass
class NormalizedDetection:
    """One detected symbol from one tile. -> converters/stage04_detection.py ->
    RawDetection (nms.py:29-52) -> DetectionRecord via the agent's own NMS + assembly.

    **CONFIRMED BY RUNNING THE REAL AGENT CODE (not documented anywhere, discovered via the
    e2e_bench smoke test):** `value` is not cosmetic. The real `build_entity`
    (stages/graph_construction/entities.py) requires a non-empty derived name/tag to
    construct a BundleEntity at all — `if not derived_name or not source_bbox: return None,
    None, suggested`. `derived_name = detection.name or _derive_clean_label(detection)`,
    and `_derive_clean_label` needs grammar-reconstructed tags, Tag ID attributes, or a
    single-token raw `value` — with none of those, the entity is SILENTLY DROPPED before
    ever reaching stage 6/11/13/12, no error, no log the converter sees. A detection with
    `value=None` and no OCR-word association is a real entity that will vanish. Molmo2's
    native output (points only, no text) therefore produces ZERO usable entities on its own
    — it needs pairing with an OCR-tag-matching step (the v1 non-goal we skipped, §8) or a
    substitute value source before it can contribute anything past stage 4 in a real
    end-to-end run. Flag this explicitly to whoever designs the harness."""
    bbox_tile: list  # [x0, y0, x1, y1] ints, TILE-LOCAL (pre-upscale-undo if applicable)
    confidence: float  # never 0.0 (D3) - assembly drops confidence==0.0 records
    entity_type: str  # MUST be one of e2e_bench.ontology.entity_types()
    value: Optional[str] = None
    entity_subtype: Optional[str] = None
    description: Optional[str] = None
    source_word_indices: list = field(default_factory=list)  # indices into that page's OCR word list


@dataclass
class NormalizedTitleBlock:
    """-> converters/stage02_titleblock.py -> TitleBlockRecord (models/title_block.py:59-82).
    D4: `located=False` is a legitimate, safe answer (stage 3 falls back to full-page
    tiling on a missing/invalid bbox - stages/tile_segmentation/exclusion.py:28-75)."""
    located: bool
    fields: dict  # keys: D5 benchmark field names (drawing_number/revision/title/site) -> str|None
    bbox_drawing_xyxy: Optional[list] = None  # converted to xywh inside the converter


@dataclass
class NormalizedSkidMember:
    target_temp_id: str
    forward_relation_name: Optional[str]  # None = "not in this skid"; else must be an
                                           # ontology relation name (benchmark: "Installed Valves")
    confidence: float
    reasoning: str = ""


@dataclass
class NormalizedSkidAssignment:
    """-> converters/stage105_skid.py -> one entry in skid_groups.json["groups"][asset_temp_id]."""
    asset_temp_id: str
    members: list  # list[NormalizedSkidMember]


@dataclass
class NormalizedEntityVerdict:
    """-> converters/stage13_entity_validation.py. keep=False maps to removed_temp_ids +
    a confidence-0.0 reclassification, per the real stage-13 write semantics
    (entity_validation/driver.py:306-333)."""
    temp_id: str
    keep: bool
    confidence: float
    reasoning: str = ""


@dataclass
class NormalizedRelationVerdict:
    """-> converters/stage12_relation_validation.py -> RelationValidation
    (models/relation_validation.py:19-40)."""
    relation_id: str
    verdict: str  # "confirmed" | "rejected" | "uncertain"
    revised_confidence: float
    reasoning: str = ""
    suggested_alternative_relation_name: Optional[str] = None
