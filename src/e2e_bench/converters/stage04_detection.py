"""
Stage 4 symbol detection converter (Conversion_Layer_Plan.md §5.4 — "the big one").

Pipeline: NormalizedDetections per tile -> RawDetection (nms.py) -> dedup_across_tiles
(the agent's real cross-tile NMS) -> _compose_detection_records (the agent's real
DetectionRecord assembly, incl. OCR-anchor bbox correction) -> Stage04Output, written to
stage-04/stage_04_output.json.

v1 non-goal (Conversion_Layer_Plan.md §8): no OCR-word<->detection association matching —
`associations=[]` always. This costs the agent's anchor-correction some accuracy; note as a
known simplification in run metadata, revisit only if detection-stage attribution flags it.
"""
from pnid_agent.models.detections import PageDetections, Stage04Output
from pnid_agent.models.drawing_document import DrawingDocument
from pnid_agent.models.page_ocr import OcrWord
from pnid_agent.shared.coord_ops import tile_to_drawing
from pnid_agent.storage.base import ArtifactStore
from pnid_agent.sub_agents.symbol_detection.driver import _compose_detection_records
from pnid_agent.sub_agents.symbol_detection.nms import RawDetection, dedup_across_tiles

from ..ontology import entity_type_names


class TileBatch:
    """One tile's worth of model output, plus enough geometry to project back to page
    coords. `origin_xy`: the tile's top-left corner IN PAGE PIXELS (pre-upscale). `upscale`:
    the factor the tile was resized by before the model saw it (1 for no upscale, 2 for the
    Molmo2 512-tile config) — detections are divided by this BEFORE the page-origin offset
    is added, matching the Stage 4 detection notebook's remap math."""
    def __init__(self, tile_index: int, origin_xy, upscale: float, detections: list):
        self.tile_index = tile_index
        self.origin_xy = origin_xy
        self.upscale = upscale
        self.detections = detections  # list[NormalizedDetection], bbox_tile in POST-upscale pixels


def convert_detection(
    *,
    drawing_document: DrawingDocument,
    artifact_store: ArtifactStore,
    page_index: int,
    tile_batches: list,  # list[TileBatch]
    ocr_words_for_page: list,  # list[NormalizedWord], SAME order as stage-01.5 for this page
    model_version: str,
    default_iou: float = 0.5,
    duration_ms: int = 0,
    cost_usd: float = 0.0,
):
    raw_per_tile = []
    tile_local_ids_per_tile = []
    for batch in tile_batches:
        raws = []
        ids = []
        for det_idx, nd in enumerate(batch.detections):
            scale = 1.0 / batch.upscale
            unscaled_tile_bbox = [c * scale for c in nd.bbox_tile]
            bbox_drawing = tile_to_drawing(unscaled_tile_bbox, tile_origin=batch.origin_xy)
            tile_local_id = f"p{page_index}_t{batch.tile_index:03d}_s{det_idx:02d}"
            raws.append(RawDetection(
                entity_type=nd.entity_type,
                entity_type_name=None,  # filled by entity_type_name_lookup in compose
                confidence=nd.confidence,
                bbox_drawing=list(bbox_drawing),
                value=nd.value,
                attributes={},
                source_word_indices=list(nd.source_word_indices),
                raw_tile_id=tile_local_id,
                raw_tile_bbox=[int(c) for c in nd.bbox_tile],  # forensics: AS SEEN by the model, post-upscale
                library_hint_class_id=None,
                entity_subtype=nd.entity_subtype,
                description=nd.description,
            ))
            ids.append(tile_local_id)
        raw_per_tile.append(raws)
        tile_local_ids_per_tile.append(ids)

    nms_result = dedup_across_tiles(
        raw_per_tile, tile_local_ids_per_tile=tile_local_ids_per_tile,
        page_index=page_index, default_iou=default_iou,
    )

    page_ocr_words = [
        OcrWord(text=w.text, bbox=list(w.bbox), confidence=w.confidence)
        for w in ocr_words_for_page
    ]

    detection_records, dropped = _compose_detection_records(
        nms_result=nms_result,
        associations=[],  # v1 non-goal: no word<->detection association matching
        page_words=page_ocr_words,
        page_index=page_index,
        entity_type_name_lookup=entity_type_names(),
        model_version=model_version,
    )

    page_detections = PageDetections(page_index=page_index, detections=detection_records)
    output = Stage04Output(
        doc_id=drawing_document.doc_id,
        model_version=model_version,
        total_duration_ms=duration_ms,
        total_cost_usd=cost_usd,
        pages_processed=[page_index],
        per_page_detection_counts={page_index: len(detection_records)},
        pages=[page_detections],
    )
    artifact_store.write_json(
        drawing_document.job_id, "stage-04/stage_04_output.json",
        output.model_dump(mode="json"),
    )
    return output, dropped
