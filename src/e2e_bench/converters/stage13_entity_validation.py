"""
Stage 13 entity validation converter (Conversion_Layer_Plan.md §5.6).

Option (a) confirmed in PHASE0_REPORT.md: calls the REAL `stage_13_run` with a
`FakeMessagesClient` standing in for Anthropic. Stage 13 makes ONE call PER PAGE (not per
entity) — the tool payload covers every entity on that page in one shot
(`reclassifications[]`, `attribute_corrections[]`, `confirmed_ok[]`, `removed_temp_ids[]`,
entity_validation/tool_schema.py:17-124) — so, unlike stage 12, no per-call content
correlation is needed: this converter always returns the SAME aggregated payload built from
all `verdicts` passed in.

keep=False maps to `removed_temp_ids` (real write semantics:
entity_validation/driver.py:306-333 — removed entities are deleted; their relations are
KEPT but stamped review_status="rejected").
"""
from pnid_agent.models.drawing_document import DrawingDocument
from pnid_agent.storage.base import ArtifactStore

from ..assembly.fake_llm import FakeMessagesClient, FakeRunner
from ..types import NormalizedEntityVerdict


def _build_payload(verdicts: list) -> dict:
    removed = [v.temp_id for v in verdicts if not v.keep]
    confirmed = [v.temp_id for v in verdicts if v.keep]
    return {
        "reclassifications": [],
        "attribute_corrections": [],
        "confirmed_ok": confirmed,
        "removed_temp_ids": removed,
    }


async def convert_entity_validation(
    *,
    context,
    artifact_store: ArtifactStore,
    drawing_document: DrawingDocument,
    verdicts: list,  # list[NormalizedEntityVerdict], all entities on the page being validated
    source_rive_uri: str = "stage-11/rive_ontology.json",
    model: str = "e2e-bench-fake",
):
    from pnid_agent.sub_agents.entity_validation.driver import stage_13_run

    payload = _build_payload(verdicts)

    def next_answer(_kwargs):
        return payload

    runner = FakeRunner(FakeMessagesClient(next_answer))
    return await stage_13_run(
        context, artifact_store, drawing_document=drawing_document,
        vlm_runner=runner, model=model, source_rive_uri=source_rive_uri,
        schema_factory=lambda: [],  # bypasses the token-based tenant ontology fetch
                                    # (ontology_validation/driver.py:236-240, "test path")
    )
