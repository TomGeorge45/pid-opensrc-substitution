"""
Stage 1.5 OCR converter (Conversion_Layer_Plan.md §5.2).

Writes the per-page word list to stage-01.5/intermediate/p{i}_words.json (a plain JSON list
of OcrWord.to_dict() dicts — page_ocr.py:19-53) AND the headline
stage-01.5/stage_01_5_output.json (Stage015Output). Word ORDER IS LOAD-BEARING: stage-4
association span_ids are `p{page}_w{global_idx}`, indexing into this exact list — freeze the
order at write time (§5.2), never resort.
"""
from pnid_agent.models.drawing_document import DrawingDocument
from pnid_agent.models.page_ocr import OcrWord, PageOcrRecord, Stage015Output
from pnid_agent.storage.base import ArtifactStore

from ..types import NormalizedWord


def convert_ocr(
    *,
    drawing_document: DrawingDocument,
    artifact_store: ArtifactStore,
    page_index: int,
    words: list,  # list[NormalizedWord], IN THE FINAL ORDER — this becomes global_idx order
    engine_name: str,  # e.g. "paddleocr" — overrides the Stage015Output default (google_vision)
    model_version: str,
    duration_ms: int = 0,
    cost_usd: float = 0.0,
) -> Stage015Output:
    words_uri = f"stage-01.5/intermediate/p{page_index}_words.json"

    ocr_words = [
        OcrWord(text=w.text, bbox=list(w.bbox), confidence=w.confidence).to_dict()
        for w in words
    ]
    artifact_store.write_json(drawing_document.job_id, words_uri, ocr_words)

    record = PageOcrRecord(
        page_index=page_index, model_version=model_version, duration_ms=duration_ms,
        cost_usd=cost_usd, n_words=len(words), words_uri=words_uri,
    )
    output = Stage015Output(
        doc_id=drawing_document.doc_id,
        ocr_engine=engine_name,
        model_version=model_version,
        total_duration_ms=duration_ms,
        total_cost_usd=cost_usd,
        pages_ocred=[page_index],
        pages_skipped=[],
        records=[record],
        per_page_word_counts={page_index: len(words)},
    )
    artifact_store.write_json(
        drawing_document.job_id, "stage-01.5/stage_01_5_output.json",
        output.model_dump(mode="json"),
    )
    return output


def word_span_id(page_index: int, global_idx: int) -> str:
    """The span_id format stage-4 associations use to reference an OCR word
    (detector.py:491, tile_words.py:25-36)."""
    return f"p{page_index}_w{global_idx}"
