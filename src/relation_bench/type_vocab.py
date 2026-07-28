"""Gap #7 — frozen PID2Graph-class -> pipeline-type-vocabulary mapping.

PID2Graph's 5 symbol classes (`pid2graph_gt.SYMBOL_CLASSES`) have no inherent relationship
to any pipeline's own type vocabulary — they're just what PID2Graph's authors happened to
name their annotation classes. Feeding a raw, un-mapped class name into a pipeline's own
type field is a real bias risk, not a cosmetic detail: e.g. prod's `hierarchy.py` only
builds a "system_member" relation when `tag.type == "equipment"` (verbatim string match,
`pnid_pipeline/hierarchy.py` line 245) — an un-mapped `"general"` would silently make that
pass never fire for every injected entity, which would look like "the LLM didn't reason
about system membership" when the real cause is "we never told it this was equipment."

FROZEN once arms run — do not change silently (same discipline as pid2graph_gt.py's
contraction rules). **Flagged for Tom's review per the register — a bad mapping could
bias the pipeline-1/2 arms specifically.**

Only pipelines 1/2 need this: their real `Tag.type` taxonomy (`pnid_pipeline/agentic.py::
ASSET_TYPES`, confirmed by direct code read) is `{instrument, valve, equipment, line,
fitting, damper, panel, room, nozzle, term_point, gauge, actuator}` (+ default "other").
Pipeline 3 (R2a/R2b, this benchmark's own port) was deliberately designed to consume
PID2Graph's classes DIRECTLY (`graph_construction/build_relations.py` doesn't touch entity
type at all — topology-only; `hierarchy/priors.py`'s EQUIP_CLASSES/CHILD_CLASSES already
use PID2Graph's own class names) — no mapping needed or wanted there, so none is provided.
"""
from __future__ import annotations

# PID2Graph class -> pnid-extraction-agent Tag.type. Rationale per mapping:
#   valve           -> "valve"       exact vocabulary match.
#   instrumentation -> "instrument"  PID2Graph's class name, extraction-agent's noun form.
#   tank            -> "equipment"   a vessel; extraction-agent has no dedicated tank type,
#                                    "equipment" is its general-asset bucket.
#   pump            -> "equipment"   same reasoning — no dedicated pump type in ASSET_TYPES.
#   general         -> "equipment"   established convention this session: on real sheets,
#                                    PID2Graph's "general" class is the equipment/vessel
#                                    class (see project memory + CLAUDE.md rule 5 context).
PID2GRAPH_TO_EXTRACTION_AGENT_TYPE = {
    "valve": "valve",
    "instrumentation": "instrument",
    "tank": "equipment",
    "pump": "equipment",
    "general": "equipment",
}


def to_extraction_agent_type(pid2graph_cls: str) -> str:
    """Map a PID2Graph class to extraction-agent's Tag.type. Falls back to "other" (the
    real Tag model's own default) for any class outside the 5 frozen symbol classes —
    should never happen for GT nodes coming out of pid2graph_gt.contract, which already
    restricts SheetGT.nodes to SYMBOL_CLASSES only."""
    return PID2GRAPH_TO_EXTRACTION_AGENT_TYPE.get(pid2graph_cls, "other")
