"""
Stage 1 sheet classification converter (Conversion_Layer_Plan.md §5.1).

Writes stage-01/stage_01_output.json (Stage01Output, models/page_classification.py:43-55)
AND sets DrawingPage.page_classification on the in-memory DrawingDocument — the latter is
required (not optional): downstream stages gate per-page work on
`DrawingPage.page_classification == PID_DRAWING`, and writing only the JSON artifact is not
enough (§5.1 CRITICAL note).
"""
from pnid_agent.models.drawing_document import DrawingDocument
from pnid_agent.models.page_classification import (
    PageClassificationLabel,
    PageClassificationRecord,
    Stage01Output,
)
from pnid_agent.storage.base import ArtifactStore


def convert_classification(
    *,
    drawing_document: DrawingDocument,
    artifact_store: ArtifactStore,
    page_index: int,
    classification: PageClassificationLabel,
    confidence: float,
    model_version: str,
    duration_ms: int = 0,
    cost_usd: float = 0.0,
) -> Stage01Output:
    record = PageClassificationRecord(
        page_index=page_index,
        classification=classification,
        confidence=confidence,
        model_version=model_version,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
    )
    pages_to_process = (
        [page_index]
        if classification in (PageClassificationLabel.PID_DRAWING, PageClassificationLabel.LEGEND)
        else []
    )
    pages_skipped = [page_index] if not pages_to_process else []

    output = Stage01Output(
        doc_id=drawing_document.doc_id,
        model_version=model_version,
        total_duration_ms=duration_ms,
        total_cost_usd=cost_usd,
        applied_treat_unknown_as=PageClassificationLabel.OTHER,
        confidence_threshold_for_skip=0.0,
        classifications=[record],
        pages_to_process=pages_to_process,
        pages_skipped=pages_skipped,
    )

    artifact_store.write_json(
        drawing_document.job_id, "stage-01/stage_01_output.json",
        output.model_dump(mode="json"),
    )

    # CRITICAL: also set page_classification on the in-memory document (§5.1)
    for page in drawing_document.pages:
        if page.page_index == page_index:
            page.page_classification = classification

    return output
