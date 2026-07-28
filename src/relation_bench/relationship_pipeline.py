"""Full relationship-extraction pipeline over GIVEN entities (isolated mode) — the
deterministic candidate-relation half that runs BEFORE any LLM validation.

This is the piece the 2026-07-24 upgrade benchmark needs: feed the real GPT-5.5-extracted
entities (id, bbox_px, type) for a sheet as fixed input, trace the PDF's own vector geometry,
and emit symbol<->symbol candidate relations — in either the ORIGINAL config or the UPGRADED
config (Tier-1 items #2 backbone pass + #4 line-label filtering). Item #1 (off-page claim
partitioning) lives in agreement_diff.py and applies to LLM CLAIMS, not to this geometry
pass, so it isn't a toggle here.

The LLM relation-validator (R4) runs AFTER this, in the GPU notebook, consuming the
candidates this produces. Kept separate so the deterministic result stands on its own
regardless of whether R4 helps or hurts (Probe 2 says it may hurt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from entities import loose_ends_from_graph, relocate_ports_to_loose_ends
from graph_construction import build_topology_relations
from pdf_vector_extract import extract_page_vector_segments
from line_tracing import build_vector_page_graph

# Which real-drawing tag types are inline pass-throughs (the backbone pass walks THROUGH
# them to connect two assets) vs assets/endpoints (it stops at them). FROZEN, flagged for
# Tom's review — same discipline as type_vocab.py's mapping. Mirrors PID2Graph's
# {valve, instrumentation, pump} pass-through set, translated to extraction-agent Tag.type.
PASSTHROUGH_TAG_TYPES = {"valve", "instrument", "fitting", "safety_device"}

# Tag types that are never a real connection endpoint — a pipe-spec text label sitting ON a
# line, not a symbol the pipe terminates at (Tier-1 item #4). Dropped from the tracer's
# endpoint set entirely, never resolved to. Same list as agreement_diff.NON_ENDPOINT_TAG_TYPES.
NON_ENDPOINT_TAG_TYPES = {"line"}

# Equipment-class tag bboxes are text labels far smaller than the drawn symbol — inflate by
# an inch of real page space so a real pipe endpoint can resolve to them. Same constant/reason
# as agreement_diff.EQUIPMENT_BBOX_PAD_INCHES (the MBD-0100 finding).
EQUIPMENT_BBOX_PAD_INCHES = 1.0


@dataclass
class CandidateRelation:
    a: str                       # entity id
    b: str                       # entity id
    source: str                  # provenance tag from build_topology_relations
    confidence: float


def _prepare_bboxes(
    entities: List[dict], render_dpi: int, *, upgraded: bool,
) -> Tuple[List[str], List[Tuple[float, float, float, float]], Dict[str, str]]:
    """Returns (ids, bboxes, class_of) for the tracer. UPGRADED config drops line-type tags
    from the endpoint set (#4); ORIGINAL keeps every entity with a usable bbox."""
    pad = EQUIPMENT_BBOX_PAD_INCHES * render_dpi
    ids: List[str] = []
    bboxes: List[Tuple[float, float, float, float]] = []
    class_of: Dict[str, str] = {}
    for e in entities:
        etype = e.get("type")
        if upgraded and etype in NON_ENDPOINT_TAG_TYPES:
            continue  # item #4 — line-label filtering (upgraded only)
        bbox = e.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        if etype == "equipment":
            x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        ids.append(e["id"])
        bboxes.append((x0, y0, x1, y1))
        class_of[e["id"]] = etype or "other"
    return ids, bboxes, class_of


def run_relationship_pipeline(
    entities: List[dict], pdf_path: str, render_dpi: int, *, upgraded: bool,
    page_index: int = 0,
) -> List[CandidateRelation]:
    """Deterministic candidate relations from given entities + the PDF's vector geometry.

    ORIGINAL config: vector trace + R2a (Stage-11 3-pass builder), no backbone, no
    line-filter, equipment-bbox inflation on (inflation predates the Tier-1 work and is part
    of the baseline both configs share).
    UPGRADED config: adds item #2 (backbone pass — walk through PASSTHROUGH_TAG_TYPES) and
    item #4 (line-label filtering — line-type tags removed from the endpoint set).
    """
    zoom = render_dpi / 72.0
    segments, page_size = extract_page_vector_segments(pdf_path, page_index, render_scale=zoom)

    ids, bboxes, class_of = _prepare_bboxes(entities, render_dpi, upgraded=upgraded)
    graph = build_vector_page_graph(page_index, segments, bboxes, ids, page_size=page_size)

    kwargs: dict = {"junction_to_detection_id": {}}
    if upgraded:
        kwargs["symbol_class_of"] = class_of
        kwargs["passthrough_symbol_classes"] = PASSTHROUGH_TAG_TYPES

    relations = build_topology_relations(graph, ids, **kwargs)
    return [CandidateRelation(a=min(r.pair), b=max(r.pair), source=r.source,
                              confidence=r.confidence) for r in relations]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 3 v2 — symbol-first input, terminal ports, single-hop backbone
# ══════════════════════════════════════════════════════════════════════════════

# Which resolved symbol types a walk may pass THROUGH to link two assets. Narrowed from the
# v1 set: `safety_device` was REMOVED. Two reasons, one physical and one measured.
# Physical: a PSV is process-TERMINAL — you relieve TO it, flow does not continue THROUGH it
# to somewhere else — so treating it as an inline fitting invents connections that the drawing
# does not contain. Measured: 3 of the 5 wrong-endpoint failures on PX-2368-0180004-001 are
# safety devices being mis-bound already (PSV-0300A, FSV-0300C, PSV-0300C attached to
# XFMR/pumps instead of the treater), so making them pass-through would compound an existing
# error rather than fix one. Also note the 762-sheet corpus check only ever validated `valve`
# (84.6%), `instrumentation` (84.8%) and `pump` (74.7%) — `safety_device` was never validated
# at all. `fitting` is kept on a physical prior (elbows/tees/reducers genuinely are
# pass-through) but is likewise un-corpus-validated; flagged rather than silently trusted.
V2_PASSTHROUGH_TYPES = {"valve", "instrument", "fitting"}

# Cap on pass-through symbols per walk. 1 == only the corpus-validated single-fitting hop.
V2_MAX_PASSTHROUGH_HOPS = 1

# Child devices that have exactly one host, and what counts as a host. Drives the per-node
# precedence guard in build_topology_relations — see its docstring for the measurements.
V2_SINGLE_HOST_CHILD_TYPES = {"instrument", "valve", "safety_device"}
V2_HOST_TYPES = {"equipment"}

# Distance cap for pass-2 traversal edges, in inches of real page space (so it travels across
# sheets rendered at different zooms). Restores the half of Stage 11's "hop/distance caps" that
# was never ported. 0.75in is chosen from the measured distribution on PX-2368-0180004-001:
# direct edges sit at median 0px / max 50px, traversal edges at median 124px with a 1767px tail.
# At 360 dpi this admits ~270px, comfortably above every legitimate short hop while cutting the
# long transitive tail. Deliberately loose rather than tuned to kill the two known-wrong edges —
# those are the single-host rule's job, and a cap tightened until a specific pair dies would be
# fitting a constant to two data points.
V2_MAX_TRAVERSAL_GAP_INCHES = 0.75

# Hop cap for BOUNDARY paths (one end an off-page connector). Separate from the on-sheet cap of
# 8 because a boundary pipe crosses the whole sheet: measured on PX-2368-0180004-001, all 15
# pentagons DO reach a real symbol — none are broken — but they need 2-14 hops, so the inherited
# cap of 8 silently discarded 8 of them. 16 clears the measured maximum with headroom, and it
# cannot inflate edge counts because each doorway keeps only its shortest connection.
V2_TERMINAL_MAX_DEPTH = 16


def assert_no_label_endpoints(entity_set, relations: Sequence["CandidateRelation"]) -> List[str]:
    """Defensive assert replacing Tier-1 #4's active line-label filter.

    In v2 pipe-spec labels are `LabelAnnotation`s and never enter the node set at all, so a
    label can no longer BE an endpoint — the old filter is structurally redundant. Rather than
    delete the protection outright, invert it: if a label id ever shows up in a relation, that
    is a contract violation upstream, and it should fail loudly instead of quietly producing
    the `2"-245-PSIG <-> 2"-245-PSIG` junk the filter used to remove. Returns a list of
    violations (empty == clean)."""
    label_ids = {l.id for l in entity_set.labels}
    bad: List[str] = []
    for r in relations:
        for nid in (r.a, r.b):
            if nid in label_ids:
                bad.append(f"label {nid} appeared as a relation endpoint in {r.a}<->{r.b}")
    return bad


@dataclass
class V2Result:
    """Candidate relations plus the provenance needed to score them honestly."""
    relations: List[CandidateRelation]
    on_sheet: List[CandidateRelation]      # symbol <-> symbol
    boundary: List[CandidateRelation]      # symbol <-> port (one end leaves the sheet)
    entity_mode: str                        # gt_injected | hand_verified | detected
    extent_sources: Dict[str, str]          # node id -> how its extent was obtained
    violations: List[str]                    # contract/invariant problems found en route
    suppressed_port_pairs: List[tuple] = field(default_factory=list)  # port<->port, dropped
    # Every drop reason is reported, never silent — a bounded pipeline must say what it cut.
    dropped_distance_cap: List[tuple] = field(default_factory=list)
    dropped_single_host: List[tuple] = field(default_factory=list)
    dropped_far_terminal: List[tuple] = field(default_factory=list)
    dropped_child_child: List[tuple] = field(default_factory=list)  # 2026-07-28 fix
    port_stats: Dict[str, int] = field(default_factory=dict)  # snapped / unsnapped / merged
    inline_stats: Dict[str, int] = field(default_factory=dict)  # passes_through annotation


def run_relationship_pipeline_v2(
    entity_set,
    pdf_path: str,
    render_dpi: int,
    *,
    page_index: int = 0,
    enable_backbone: bool = True,
    max_path_depth: int = 8,
    relocate_ports: bool = False,
    route_inline: bool = True,
) -> V2Result:
    """Pipeline 3 v2 over an :class:`entities.EntitySet`.

    Differences from v1, all of them consequences of the input contract rather than new
    heuristics:
      - Endpoints resolve against real symbol EXTENTS and port extents, not text bboxes. The
        `EQUIPMENT_BBOX_PAD_INCHES` inflation hack is simply absent here — it was a workaround
        for name-plate boxes and measured partial-and-insufficient (MBD-0100 gained one
        connection; GD-B-540 stayed at exactly 0; PX-2365 went 0->2).
      - Ports are terminal, killing the border-column false edges by construction.
      - The backbone pass is capped at a single fitting hop.
      - Pipe labels are not in the node set at all, so they cannot be endpoints.

    Returns on-sheet and boundary relations SEPARATELY. They are never summed into one number:
    an on-sheet edge is verified by geometry alone, a boundary edge needs geometry plus a text
    read of the doorway, so they have different evidence standards — the same discipline as
    CLAUDE.md rule 5's refusal to average detection and typing.
    """
    violations = list(entity_set.validate())

    zoom = render_dpi / 72.0
    segments, page_size = extract_page_vector_segments(pdf_path, page_index, render_scale=zoom)

    # ── PASS 1 — optional port relocation, OFF BY DEFAULT ────────────────────
    # `relocate_ports` snaps each port onto the nearest loose pipe end. The machinery works, but
    # the evidence that justified it did not survive scrutiny: the original 22/22 measurement
    # was taken against a graph that included the port text boxes, and the tracer masks symbol
    # boxes, so it was measuring each text box against its own mask boundary. Unconfounded, only
    # 8 of 22 land within 108px and the median is 183px. Enabling it made boundary edges go from
    # 1 to 0. Left available for experimentation, defaulted off rather than deleted.
    port_stats: Dict[str, int] = {}
    if relocate_ports and entity_set.ports:
        s_ids, s_boxes, _ = entity_set.tracer_inputs(include_ports=False)
        pre_graph = build_vector_page_graph(
            page_index, segments, s_boxes, s_ids, page_size=page_size)
        port_stats = relocate_ports_to_loose_ends(
            entity_set, loose_ends_from_graph(pre_graph), render_dpi=render_dpi)

    # ── PASS 2 — trace against symbols + located ports ───────────────────────
    ids, bboxes, class_of = entity_set.tracer_inputs()
    graph = build_vector_page_graph(page_index, segments, bboxes, ids, page_size=page_size)

    # `passes_through_symbols` is populated by the tracer itself
    # (`vector_graph._resolve_passthrough_symbols`). `route_inline` below additionally makes those
    # inline symbols routable WAYPOINTS in the traversal graph — the tracer's field alone only lets
    # Pass 0 chain symbols sharing one segment, while an inline valve's neighbours are reached via
    # other segments through junctions. Measured effect: isolated symbols 11 -> 9.
    inline_stats: Dict[str, int] = {}

    node_bbox = {nid: bb for nid, bb in zip(ids, bboxes)}
    stats: Dict[str, object] = {}
    kwargs: dict = {
        "junction_to_detection_id": {},
        "max_path_depth": max_path_depth,
        "terminal_ids": entity_set.terminal_ids(),
        "stats_out": stats,
        "node_bbox": node_bbox,
        "symbol_class_of": class_of,
        "max_traversal_gap_px": V2_MAX_TRAVERSAL_GAP_INCHES * render_dpi,
        "single_host_child_classes": V2_SINGLE_HOST_CHILD_TYPES,
        "host_classes": V2_HOST_TYPES,
        "terminal_max_depth": V2_TERMINAL_MAX_DEPTH,
        "route_through_inline": route_inline,
    }
    if enable_backbone:
        kwargs["passthrough_symbol_classes"] = V2_PASSTHROUGH_TYPES
        kwargs["max_passthrough_hops"] = V2_MAX_PASSTHROUGH_HOPS

    relations = [
        CandidateRelation(a=min(r.pair), b=max(r.pair), source=r.source,
                          confidence=r.confidence)
        for r in build_topology_relations(graph, ids, **kwargs)
    ]

    violations.extend(assert_no_label_endpoints(entity_set, relations))

    port_ids = entity_set.terminal_ids()
    on_sheet = [r for r in relations if r.a not in port_ids and r.b not in port_ids]
    boundary = [r for r in relations if (r.a in port_ids) != (r.b in port_ids)]
    # Terminality is enforced at emit time in build_topology_relations, so nothing should
    # reach here with both ends terminal. Kept as a belt-and-braces assert: if it ever fires,
    # the contract broke upstream and should be loud, not quietly bucketed.
    for r in relations:
        if r.a in port_ids and r.b in port_ids:
            violations.append(f"port<->port relation survived terminality: {r.a}<->{r.b}")

    return V2Result(
        relations=relations,
        on_sheet=on_sheet,
        boundary=boundary,
        entity_mode=entity_set.mode,
        extent_sources=entity_set.extent_source_of(),
        violations=violations,
        suppressed_port_pairs=list(stats.get("suppressed_terminal_pairs") or []),
        dropped_distance_cap=list(stats.get("dropped_distance_cap") or []),
        dropped_single_host=list(stats.get("dropped_single_host") or []),
        dropped_far_terminal=list(stats.get("dropped_far_terminal") or []),
        dropped_child_child=list(stats.get("dropped_child_child_no_shared_host") or []),
        port_stats=dict(port_stats),
        inline_stats=dict(inline_stats),
    )
