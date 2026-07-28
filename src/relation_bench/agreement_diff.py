"""Part B — the agreement-diff itself: does GPT-5.5-low's real connectivity reasoning
(prod's `apply_hierarchy` + `build_result`'s baseline relationships) agree with what the
PDF's own vector geometry physically traces?

Honest framing, stated once here rather than re-litigated per caller (per the original Part
B plan): the traced geometry is NOT ground truth — `build_vector_page_graph` is a real
algorithm with real failure modes (bridging heuristics, degree-2 contraction, symbol-snap
radius), so this is "agreement between two independent real signals, plus human
spot-verification of the disagreements," not an absolute F1 against a trusted answer key.
Reported SYMMETRICALLY (both directions), not as one side scored against the other:
  - agreement_rate_llm       = of what the LLM claimed connected, how much does geometry
                               also show connected. Low here could mean LLM over-claims
                               (hallucinated/indirect relations) OR the tracer missed real
                               lines (its own known failure mode, see R1's PID2Graph ceiling
                               finding for how real that risk is).
  - agreement_rate_geometry  = of what geometry traced as connected, how much did the LLM
                               also claim. Low here could mean the LLM missed real
                               connections OR the tracer over-connects (a bridging/noding
                               false positive).
Neither number alone tells you which side is "right" — that's what the disagreement
overlays (rendered separately) are for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

from graph_construction import build_topology_relations
from pdf_vector_extract import extract_page_vector_segments
from line_tracing import build_vector_page_graph


@dataclass
class OffPageClaim:
    """An LLM connectivity claim where exactly one endpoint never resolved to a real
    on-sheet tag id — hierarchy.py's `aln_to_id.get(..., text)` fallback path. Gap #22:
    the Fable adjudication found ~19-20 of ~25 true claims on PX-2368-0180004-001 were
    exactly this shape (remote equipment named only inside a border connector annotation,
    not drawn on this sheet at all) — structurally unverifiable by ANY single-sheet
    tracer. Previously these were silently dropped inside `_llm_claimed_pairs` with no
    visible trace (gap #22a) — that made `agreement_rate_llm`'s denominator look like it
    covered every claim when it silently didn't. Surfaced explicitly here instead so the
    on-sheet metric is honest about its own scope, and so a future off-page checker (does
    the claim correctly name the off-page equipment the connector text references — a
    Probe-3-shaped reading task, no tracing) has a concrete, counted worklist to run
    against, not a guess at how big the off-page bucket even is."""
    on_sheet_id: str
    off_page_text: str
    kind: str


@dataclass
class AgreementResult:
    sheet_id: str
    llm_pairs: Set[FrozenSet[str]] = field(default_factory=set)
    traced_pairs: Set[FrozenSet[str]] = field(default_factory=set)
    agree: Set[FrozenSet[str]] = field(default_factory=set)
    llm_only: Set[FrozenSet[str]] = field(default_factory=set)
    geometry_only: Set[FrozenSet[str]] = field(default_factory=set)
    off_page_claims: List[OffPageClaim] = field(default_factory=list)

    @property
    def agreement_rate_llm(self) -> float:
        n = len(self.llm_pairs)
        return len(self.agree) / n if n else 0.0

    @property
    def agreement_rate_geometry(self) -> float:
        n = len(self.traced_pairs)
        return len(self.agree) / n if n else 0.0

    @property
    def on_sheet_claim_count(self) -> int:
        """Total LLM connectivity claims whose BOTH endpoints are on this sheet — the
        denominator `agreement_rate_llm` is actually computed over. Compare against
        `on_sheet_claim_count + len(off_page_claims)` for the TRUE total claim count."""
        return len(self.llm_pairs)


# Only these relation kinds are physical-connectivity claims ("a line/signal directly
# joins these two tags") — hierarchy.py's own `conn_kinds`, emitted by the dedicated
# connectivity pass. `hosted_by`/`on_line`/`system_member`/`loop_member` are hierarchy/
# containment/grouping relations (parent-child asset breakdown, loop membership) — NOT
# claims that a drawn pipe connects the two endpoints, so they are the WRONG thing to diff
# against traced line geometry. Confirmed this distinction matters empirically: diffing all
# relation kinds against traced pairs gave near-zero agreement on every sheet (0-20%) even
# though the SAME entities showed up on both sides (78/79 and 132/139 touched-id overlap) —
# the entities were right, the relation KIND being compared was wrong.
CONNECTIVITY_KINDS = {"feeds", "relieves_to", "actuates", "signal_to"}


def _llm_claimed_pairs(
    tags_by_id: Dict[str, dict], relationships: List[dict],
) -> Tuple[Set[FrozenSet[str]], List[OffPageClaim]]:
    """Partitions prod's real PHYSICAL-CONNECTIVITY relations (see CONNECTIVITY_KINDS
    docstring above) into on-sheet<->on-sheet pairs (both endpoints resolve to a real tag
    id — these are what's fairly diffable against traced geometry) vs on-sheet<->off-page
    claims (gap #22a) — exactly one endpoint resolved, the other is a raw off-sheet TEXT
    string (hierarchy.py's `aln_to_id.get(..., text)` fallback when no tag id matched).
    Claims where NEITHER endpoint resolves are dropped entirely (unscoreable either way,
    e.g. a self-referential or malformed relation)."""
    pairs: Set[FrozenSet[str]] = set()
    off_page: List[OffPageClaim] = []
    for rel in relationships:
        if rel.get("kind") not in CONNECTIVITY_KINDS:
            continue
        a, b = rel.get("from"), rel.get("to")
        if a is None or b is None or a == b:
            continue
        a_on, b_on = a in tags_by_id, b in tags_by_id
        if a_on and b_on:
            pairs.add(frozenset((a, b)))
        elif a_on and not b_on:
            off_page.append(OffPageClaim(on_sheet_id=a, off_page_text=str(b), kind=rel["kind"]))
        elif b_on and not a_on:
            off_page.append(OffPageClaim(on_sheet_id=b, off_page_text=str(a), kind=rel["kind"]))
        # else: neither resolves -- unscoreable either way, dropped
    return pairs, off_page


# Equipment-class tags (vessels, drums, exchangers) get a real drawn symbol far larger
# than their tag's text-label bbox — confirmed visually (2026-07-23 overlay on
# PX-2368-0180004-001): nearly every disagreement traced back to ONE hub vessel (MBD-0100)
# whose bbox was a tiny (130x25px) text label, sitting apart from its actual drawn ellipse
# outline, so no real pipe endpoint ever landed close enough to resolve. Valve/instrument
# tags don't need this — their labels sit at/near their (small) actual symbol already (the
# "agree" pairs were overwhelmingly these types). Padding scales with render_dpi (an inch
# of real page space, not a fixed pixel count, so it's consistent across sheets rendered at
# different zooms) — 1 inch is a deliberately generous, untuned first guess, not a
# calibrated constant; revisit if it over- or under-shoots on more sheets.
EQUIPMENT_BBOX_PAD_INCHES = 1.0


# Tag type "line" (prod's real ASSET_TYPES, e.g. a pipe-spec label like `6"(300#)`)
# is a text annotation SITTING ON a pipe, not a symbol the pipe terminates at — including
# it as a tracer endpoint lets the pipe "connect to its own label" or split a single real
# run into two spurious segments at the label's position. Confirmed as a real noise
# source in the Fable adjudication's geometry_only pairs (equipment<->line-LABEL pairs
# the LLM correctly never claims as equipment endpoints). Dropped here entirely — never
# added to symbol_bboxes/symbol_det_ids, same "masked but never a valid endpoint"
# convention `process_page.py`'s `extra_mask_bboxes` already uses for text.
NON_ENDPOINT_TAG_TYPES = {"line"}


def _inflate_equipment_bboxes(tags_by_id: Dict[str, dict], render_dpi: int) -> Dict[str, Tuple[float, float, float, float]]:
    pad = EQUIPMENT_BBOX_PAD_INCHES * render_dpi
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for tid, t in tags_by_id.items():
        if t.get("type") in NON_ENDPOINT_TAG_TYPES:
            continue
        bbox = t.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        if t.get("type") == "equipment":
            x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        out[tid] = (x0, y0, x1, y1)
    return out


def _traced_pairs(tags_by_id: Dict[str, dict], pdf_path: str, render_dpi: int) -> Set[FrozenSet[str]]:
    """Vector-traced symbol<->symbol pairs, using the SAME real tag bboxes/ids the LLM saw
    (so both sides of the diff are grounded in the same entity set) and the real Stage-11
    3-pass topology builder (inline-chain / direct-endpoint / hop-capped BFS), same as
    pipeline 3's R2a — no ontology-naming needed here either, topology existence only.
    Equipment-class tag bboxes are inflated first — see EQUIPMENT_BBOX_PAD_INCHES above."""
    zoom = render_dpi / 72.0
    segments, page_size = extract_page_vector_segments(pdf_path, 0, render_scale=zoom)

    inflated = _inflate_equipment_bboxes(tags_by_id, render_dpi)
    ids = list(inflated.keys())
    bboxes = [inflated[tid] for tid in ids]

    graph = build_vector_page_graph(0, segments, bboxes, ids, page_size=page_size)
    relations = build_topology_relations(graph, ids, junction_to_detection_id={})
    return {r.pair for r in relations}


def compute_agreement(sheet_id: str, extraction_result: dict, pdf_path: str) -> AgreementResult:
    """`extraction_result` is the dumped `ExtractionResult` JSON (as saved by
    `run_real_extraction_partB.py`): needs `tags` (id, bbox_px) and `relationships`
    (from/to) at minimum, plus `drawing.render_dpi` to recover the render zoom the tag
    bboxes were measured in."""
    tags_by_id = {t["id"]: t for t in extraction_result["tags"]}
    render_dpi = extraction_result["drawing"]["render_dpi"]

    llm_pairs, off_page_claims = _llm_claimed_pairs(
        tags_by_id, extraction_result.get("relationships", []))
    traced_pairs = _traced_pairs(tags_by_id, pdf_path, render_dpi)

    agree = llm_pairs & traced_pairs
    return AgreementResult(
        sheet_id=sheet_id,
        llm_pairs=llm_pairs,
        traced_pairs=traced_pairs,
        agree=agree,
        llm_only=llm_pairs - traced_pairs,
        geometry_only=traced_pairs - llm_pairs,
        off_page_claims=off_page_claims,
    )


def format_agreement(result: AgreementResult) -> str:
    total_claims = result.on_sheet_claim_count + len(result.off_page_claims)
    off_page_pct = (len(result.off_page_claims) / total_claims) if total_claims else 0.0
    return (
        f"{result.sheet_id}: on-sheet claims={len(result.llm_pairs)} "
        f"off-page claims={len(result.off_page_claims)} ({off_page_pct:.0%} of {total_claims} total) "
        f"traced={len(result.traced_pairs)} agree={len(result.agree)} "
        f"(agreement_rate_llm={result.agreement_rate_llm:.3f}, "
        f"agreement_rate_geometry={result.agreement_rate_geometry:.3f}) "
        f"llm_only={len(result.llm_only)} geometry_only={len(result.geometry_only)}"
    )
