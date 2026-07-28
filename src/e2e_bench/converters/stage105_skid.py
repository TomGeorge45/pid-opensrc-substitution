"""
Stage 10.5 skid grouping converter (Conversion_Layer_Plan.md §5.5).

D8: writes the sidecar JSON directly (`stage-10.5/skid_groups.json`) rather than running
the real `sub_agents/skid_grouping/driver.py` (which needs a `vlm_runner` + rive_adk
wiring for its own per-asset ROI-crop LLM loop) — the sidecar IS the interface consumed by
`infer_from_skid_groups` (stages/graph_construction/inference.py:705-726), confirmed by
that function's docstring in the real driver: "Stage 11 picks this file up via
infer_from_skid_groups and emits one BundleRelation per assignment whose
forward_relation_name is not None."
"""
from pnid_agent.models.drawing_document import DrawingDocument
from pnid_agent.storage.base import ArtifactStore

from ..types import NormalizedSkidAssignment


def convert_skid_groups(
    *,
    drawing_document: DrawingDocument,
    artifact_store: ArtifactStore,
    pages_processed: list,
    assignments: list,  # list[NormalizedSkidAssignment]
    total_cost_usd: float = 0.0,
    total_duration_ms: int = 0,
) -> dict:
    groups = {}
    for a in assignments:
        groups[a.asset_temp_id] = [
            {
                "target_temp_id": m.target_temp_id,
                "forward_relation_name": m.forward_relation_name,
                "confidence": m.confidence,
                "reasoning": m.reasoning,
            }
            for m in a.members
        ]

    payload = {
        "schema_version": "1.0",
        "doc_id": drawing_document.doc_id,
        "pages_processed": list(pages_processed),
        "groups": groups,
        "telemetry": {
            "total_cost_usd": total_cost_usd,
            "total_duration_ms": total_duration_ms,
            "per_asset": [],
        },
    }
    artifact_store.write_json(
        drawing_document.job_id, "stage-10.5/skid_groups.json", payload,
    )
    return payload
