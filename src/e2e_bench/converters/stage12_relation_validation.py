"""
Stage 12 relation validation converter (Conversion_Layer_Plan.md §5.7).

Option (a) confirmed in PHASE0_REPORT.md: calls the REAL `stage_12_run` with a
`FakeMessagesClient`. UNLIKE stage 13, stage 12 dispatches concurrently via
`asyncio.gather(*(_one(r) for r in to_validate))`, and each `client.messages.create(...)`
call is about exactly ONE relation — but the prompt text (relation_validation/prompt.py
build_user_message) contains NO relation_id, only `source_label`/`target_label`/
`relation_name`/`pipeline_confidence` (confirmed by reading the prompt source). So this
converter correlates each incoming call to the right precomputed verdict by matching that
same tuple, extracted from `kwargs["messages"]`'s text content via regex.

**Known limitation (real, not hypothetical):** if two relations on the same page share an
identical (source_label, target_label, relation_name, pipeline_confidence) tuple — e.g. two
unlabeled entities ("(unlabeled)") connected by the same relation name at the same
confidence — this lookup is ambiguous and picks whichever matches first. This is a ceiling
imposed by the real prompt's content, not something fixable without modifying agent code
(out of scope per plan D1). Rare in practice (entity names are usually unique per sheet)
but worth flagging if relation-validation results ever look systematically wrong for
sheets with many unlabeled entities.
"""
import re

from pnid_agent.models.drawing_document import DrawingDocument
from pnid_agent.storage.base import ArtifactStore

from ..assembly.fake_llm import FakeMessagesClient, FakeRunner
from ..types import NormalizedRelationVerdict

_FIELD_RE = re.compile(
    r"source_label: (?P<source_label>.*)\n"
    r"target_entity_type: .*\n"
    r"target_label: (?P<target_label>.*)\n"
    r"relation_name: (?P<relation_name>.*)\n"
    r"detected_line_type: .*\n"
    r"candidate_relation_names: .*\n"
    r"pipeline_confidence: (?P<pipeline_confidence>[0-9.]+)"
)


def _extract_message_text(kwargs: dict) -> str:
    for msg in kwargs.get("messages", []):
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
    return ""


def _correlation_key(source_label, target_label, relation_name, pipeline_confidence):
    return (source_label.strip(), target_label.strip(), relation_name.strip(),
            round(float(pipeline_confidence), 3))


def _build_answer_lookup(verdicts: list, relations_by_id: dict, entities_by_temp_id: dict):
    """verdicts: list[NormalizedRelationVerdict]. relations_by_id: {relation_id: BundleRelation}
    (needed to reconstruct the same correlation key the real prompt would produce)."""
    lookup = {}
    for v in verdicts:
        rel = relations_by_id[v.relation_id]
        src = entities_by_temp_id.get(rel.source_temp_id)
        tgt = entities_by_temp_id.get(rel.target_temp_id)
        source_label = (src.name if src and src.name else "(unlabeled)")
        target_label = (tgt.name if tgt and tgt.name else "(unlabeled)")
        key = _correlation_key(source_label, target_label, rel.forward_relation_name, rel.confidence)
        lookup[key] = {
            "verdict": v.verdict,
            "revised_confidence": v.revised_confidence,
            "reasoning": v.reasoning,
            "suggested_alternative_relation_name": v.suggested_alternative_relation_name,
            "annotations": {},
        }
    return lookup


async def convert_relation_validation(
    *,
    context,
    artifact_store: ArtifactStore,
    drawing_document: DrawingDocument,
    verdicts: list,  # list[NormalizedRelationVerdict]
    relations_by_id: dict,  # {relation_id: BundleRelation} for every relation being validated
    entities_by_temp_id: dict,  # {temp_id: BundleEntity} for source/target label lookup
    confidence_threshold: float = 0.75,
    source_rive_uri: str = "stage-11/rive_ontology.json",
    model: str = "e2e-bench-fake",
):
    from pnid_agent.sub_agents.relation_validation.driver import stage_12_run

    lookup = _build_answer_lookup(verdicts, relations_by_id, entities_by_temp_id)

    def next_answer(kwargs):
        text = _extract_message_text(kwargs)
        m = _FIELD_RE.search(text)
        if not m:
            raise RuntimeError(f"stage12 fake client: could not parse correlation fields from prompt text: {text[:300]!r}")
        key = _correlation_key(**m.groupdict())
        if key not in lookup:
            raise RuntimeError(f"stage12 fake client: no precomputed verdict for key {key}")
        return lookup[key]

    runner = FakeRunner(FakeMessagesClient(next_answer))
    return await stage_12_run(
        context, artifact_store, drawing_document=drawing_document,
        vlm_runner=runner, model=model, confidence_threshold=confidence_threshold,
        source_rive_uri=source_rive_uri,
    )
