"""
Stage 2 title block converter (Conversion_Layer_Plan.md §5.3).

D4: `located=False, bbox_drawing=None` is a legitimate, safe answer — stage 3 falls back to
full-page tiling on a missing/invalid bbox (stages/tile_segmentation/exclusion.py:28-75).
D5: `fields` keys are the benchmark's declared 4-field tenant schema
(drawing_number/revision/title/site), not a real fixed list (none exists in code — the
real keys are tenant-runtime, title_block_extraction/driver.py:304).

NOTE: TitleBlockRecord.bbox_drawing is xywh (models/title_block.py:66-69) — everywhere
else in this codebase bboxes are xyxy. NormalizedTitleBlock.bbox_drawing_xyxy is xyxy at
the normalized-type level for consistency (types.py); this converter does the xyxy->xywh
conversion right at the boundary, per D9/types.py's docstring.
"""
from pnid_agent.models.drawing_document import DrawingDocument
from pnid_agent.models.provenance import AttributeProvenance
from pnid_agent.models.title_block import Stage02Output, TitleBlockField, TitleBlockRecord
from pnid_agent.storage.base import ArtifactStore

from ..types import NormalizedTitleBlock


def _xyxy_to_xywh(bbox_xyxy):
    x0, y0, x1, y1 = bbox_xyxy
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def convert_titleblock(
    *,
    drawing_document: DrawingDocument,
    artifact_store: ArtifactStore,
    page_index: int,
    normalized: NormalizedTitleBlock,
    ocr_engine: str,
    source: str,  # e.g. "e2e_bench_qwen" / "e2e_bench_gpt55" (AttributeProvenance.source)
    field_confidence: float = 0.5,
    duration_ms: int = 0,
    cost_usd: float = 0.0,
) -> Stage02Output:
    fields = {}
    for key, value in normalized.fields.items():
        if value is None:
            continue
        fields[key] = TitleBlockField(
            value=value,
            provenance=AttributeProvenance(confidence=field_confidence, source=source),
        )

    bbox_drawing = (
        _xyxy_to_xywh(normalized.bbox_drawing_xyxy)
        if normalized.bbox_drawing_xyxy is not None else None
    )

    record = TitleBlockRecord(
        page_index=page_index,
        located=normalized.located,
        bbox_drawing=bbox_drawing,
        fields=fields,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
        vlm_used=True,
    )

    output = Stage02Output(
        doc_id=drawing_document.doc_id,
        ocr_engine=ocr_engine,
        total_duration_ms=duration_ms,
        total_cost_usd=cost_usd,
        pages_with_title_block=[page_index] if normalized.located else [],
        pages_without_title_block=[] if normalized.located else [page_index],
        records=[record],
    )
    artifact_store.write_json(
        drawing_document.job_id, "stage-02/title_block_output.json",
        output.model_dump(mode="json"),
    )
    return output
