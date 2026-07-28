"""R2a — deterministic connectivity builder, ported from pnid-intelligence-agent
Stage 11 (agents/pnid-intelligence-agent/pnid_agent/stages/graph_construction/relations.py
``build_relations``). Same three-pass strategy, same precedence rules, same hop-depth cap
via ``path_traversal.LineGraphAdjacency`` — the actual "Stage 11 + hop/distance caps" work
item from Benchmark_Gaps_Register.md gap #10.

What's INTENTIONALLY stripped vs. the real ``build_relations``:
  - Ontology relation-name resolution (``OntologyRelationIndex`` / ``BundleRelation`` /
    ``_resolve_ontology_relations``). Real Stage 11 looks up the tenant's ontology to name
    each relation (``feeds_into``, ``operates``, etc.) and can refuse to emit a relation
    when no ontology row matches a (src_type, tgt_type) pair. This benchmark scores
    TOPOLOGY ONLY — does an edge exist — never relation naming/kind (CLAUDE.md: never
    average detection/typing; the analogous rule here is never gate topology existence on
    a typed-relation vocabulary lookup we have no reference ontology for, per the
    still-open ontology-coverage question in CLAUDE.md's rule 5). So every candidate pair
    the three passes find is emitted as a topology edge unconditionally.
  - The ISA valve-operator authority rule (``isa_rules.operates_valve_authority``) is NOT
    wired in here. That rule parses ISA tag TEXT, and PID2Graph nodes carry no tag text at
    all (class-agnostic, like Gupta) — the rule is real and portable but genuinely inert on
    this dataset. It stays un-vendored until the AG/RIVE annotated fixture (Group 2) exists
    to actually test it against; vendoring its ~650-line grammar-parser dependency now
    would be speculative work with nothing to run it on.

Passes (identical structure/order to the source, precedence: pass 0/1 direct evidence
beats pass 2 traversal for the same pair):
  0. Inline-chain — consecutive symbols a single pipe run physically passes through
     (``segment.passes_through_symbols``).
  1. Direct connections — segment endpoints that resolve straight to a symbol (including
     a junction sitting exactly at a symbol's mask boundary).
  2. Graph traversal — BFS through junctions up to ``max_path_depth`` hops, deduped to the
     shortest path per (source, target) pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from line_tracing.models import PageLineGraph

from .path_confidence import compute_path_confidence, dominant_line_type
from .path_traversal import LineGraphAdjacency


@dataclass(frozen=True)
class TopologyRelation:
    pair: FrozenSet[str]          # {detection_id_a, detection_id_b}
    source: str                  # provenance tag, e.g. "line_segment_p0_g0012"
    line_type: str
    confidence: float


def _bbox_gap(a: Optional[tuple], b: Optional[tuple]) -> Optional[float]:
    """Shortest gap between two boxes, 0.0 when they touch or overlap. Same measure the
    precision worklist reports, so the distance cap and the audit talk about the same number."""
    if not a or not b or len(a) < 4 or len(b) < 4:
        return None
    dx = max(0.0, max(b[0] - a[2], a[0] - b[2]))
    dy = max(0.0, max(b[1] - a[3], a[1] - b[3]))
    return (dx * dx + dy * dy) ** 0.5


def _effective_detection_id(endpoint, junction_to_detection_id: Dict[str, str]) -> Optional[str]:
    if endpoint.kind == "symbol" and endpoint.ref:
        return str(endpoint.ref)
    if endpoint.kind == "junction" and endpoint.ref:
        return junction_to_detection_id.get(str(endpoint.ref))
    return None


def build_topology_relations(
    page_graph: PageLineGraph,
    valid_detection_ids: Sequence[str],
    *,
    enable_path_traversal: bool = True,
    max_path_depth: int = 8,
    junction_to_detection_id: Optional[Dict[str, str]] = None,
    symbol_class_of: Optional[Dict[str, str]] = None,
    passthrough_symbol_classes: Optional[Set[str]] = None,
    terminal_ids: Optional[Set[str]] = None,
    max_passthrough_hops: Optional[int] = None,
    stats_out: Optional[Dict[str, object]] = None,
    node_bbox: Optional[Dict[str, tuple]] = None,
    max_traversal_gap_px: Optional[float] = None,
    single_host_child_classes: Optional[Set[str]] = None,
    host_classes: Optional[Set[str]] = None,
    terminal_max_depth: Optional[int] = None,
    route_through_inline: bool = False,
) -> List[TopologyRelation]:
    """Topology-only port of Stage 11's ``build_relations``. ``valid_detection_ids`` is
    the set of GT symbol detection_ids (from pid2graph_gt.SheetGT.nodes) — a segment
    endpoint resolving to anything outside this set (shouldn't happen when R1 was fed the
    same symbol_bboxes/symbol_det_ids, but checked defensively) is dropped.

    ``symbol_class_of``/``passthrough_symbol_classes`` (gap #14, the process-backbone
    pass): optional class map + pass-through-eligible class set (e.g. PID2Graph's
    ``{"valve", "instrumentation", "pump"}`` — everything except the asset/equipment
    classes ``general``/``tank``) so pass 2's BFS traversal can walk THROUGH inline
    fittings to reach the next asset, not just stop at the first one. Additive only —
    when omitted, behavior is byte-identical to before (every existing self-tested number
    is unaffected).

    ``terminal_ids``/``max_passthrough_hops`` (Pipeline 3 v2): forwarded to
    ``find_symbol_paths`` — ids that may be walked TO but never THROUGH (off-page connector
    ports), and a cap on pass-through symbols per walk. See ``path_traversal`` for the
    measurements motivating both. Also additive; omitting them preserves prior behavior.

    ``max_traversal_gap_px`` — the DISTANCE cap. This module's own header has always claimed to
    implement "Stage 11 + hop/distance caps", but only the hop cap (``max_path_depth``) was
    ever ported; nothing rejected a long path. ``path_confidence`` applies a length *penalty*,
    which lowers a score but never drops an edge. The register credits Stage 11's caps with
    taking relations 91 -> 20 by removing "spurious transitive links", so we had half the
    mechanism and specifically the half that discards nothing. Applies to pass-2 traversal
    edges only — pass 0/1 are direct physical evidence and are never distance-gated.

    ``single_host_child_classes``/``host_classes`` — per-NODE precedence, the threshold-free
    guard. Existing precedence is per-PAIR ("pass 0/1 direct evidence beats pass 2 traversal
    for the same pair"), which cannot catch the measured failure: LSHL-0100 holds a direct
    gap-0 edge to the vessel AND a 5-segment traversal edge to HAM-0100 — different pairs, so
    no conflict is detected, and the wrong one survives. An instrument has exactly ONE host by
    ISA convention, which is why the delta adjudication classified these as "WRONG ENDPOINT"
    rather than "extra". So: once a child device has direct evidence of a host, its
    traversal-derived edges to OTHER hosts are dropped.
    Measured on PX-2368-0180004-001: direct edges have median gap 0px (max 50px) while
    traversal edges have median 124px (max 1767px); the two correct vessel edges are gap-0
    direct, the two wrong HAM edges are 110px/155px traversal.
    Recall is preserved by construction — a child with ONLY traversal evidence keeps it, since
    that is the only hypothesis available. Equipment<->equipment traversal is untouched, so the
    vessel-through-valve backbone case still fires."""
    valid_ids = set(valid_detection_ids)
    j2d: Dict[str, str] = junction_to_detection_id or {}
    terminal = terminal_ids or set()
    emitted_pairs: Set[Tuple[str, str]] = set()
    out: List[TopologyRelation] = []
    suppressed_terminal_pairs: List[Tuple[str, str]] = []

    def _emit(a: str, b: str, source: str, line_type: str, confidence: float) -> None:
        if a not in valid_ids or b not in valid_ids or a == b:
            return
        # Terminal<->terminal (port<->port) is suppressed in EVERY pass, not just traversal.
        # Blocking it only in pass 2's BFS is insufficient: passes 0 and 1 emit direct
        # endpoint pairs, and the border-column artifact reaches them that way — a border
        # frame line or annotation leader running down the connector column snaps its two ends
        # to two adjacent doorway boxes and emits the edge with no traversal involved.
        # Confirmed empirically on PX-2368-0180004-001 (2026-07-27): with traversal-only
        # terminality, 10 port<->port edges still came through, all consecutive border-column
        # entries (t0039<->t0040, t0041<->t0042, t0044<->t0046, ...) — the same failure family
        # as the backbone pass's measured `MBF-0623`<->`HBG-0905`.
        # A pipe that genuinely crosses the sheet from one doorway to another IS physically
        # real, but it is a relation between two OTHER sheets mediated by this one, not a
        # symbol<->symbol relation here — nothing this benchmark can score, and
        # indistinguishable from the artifact anyway. Counted rather than silently dropped.
        if a in terminal and b in terminal:
            suppressed_terminal_pairs.append((a, b))
            return
        key = tuple(sorted((a, b)))
        if key in emitted_pairs:
            return
        emitted_pairs.add(key)
        out.append(TopologyRelation(pair=frozenset((a, b)), source=source,
                                     line_type=line_type, confidence=confidence))

    # ── Pass 0 — inline-chain through passes_through_symbols ────────────────────
    for segment in page_graph.segments:
        passed = list(segment.passes_through_symbols or [])
        if not passed:
            continue
        chain: List[str] = []
        if segment.endpoint_a.kind == "symbol" and segment.endpoint_a.ref:
            chain.append(str(segment.endpoint_a.ref))
        chain.extend(str(d) for d in passed)
        if segment.endpoint_b.kind == "symbol" and segment.endpoint_b.ref:
            chain.append(str(segment.endpoint_b.ref))
        for src, tgt in zip(chain, chain[1:]):
            _emit(src, tgt, f"passes_through_{segment.segment_id}",
                  segment.line_type, round(float(segment.confidence), 4))

    # ── Pass 1 — direct segment endpoints (incl. junction-at-symbol-boundary) ───
    for segment in page_graph.segments:
        if segment.endpoint_a.kind == "loose_end" or segment.endpoint_b.kind == "loose_end":
            continue
        src_det = _effective_detection_id(segment.endpoint_a, j2d)
        tgt_det = _effective_detection_id(segment.endpoint_b, j2d)
        if src_det is None or tgt_det is None:
            continue
        _emit(src_det, tgt_det, f"line_segment_{segment.segment_id}",
              segment.line_type, round(float(segment.confidence), 4))

    # ── Pass 2 — BFS graph traversal, hop-depth + distance capped ───────────────
    # Snapshot which child devices already have DIRECT evidence of a host, before any
    # traversal edge is emitted. Must be taken here, after passes 0/1 and before pass 2, or the
    # rule would see its own output.
    classes = symbol_class_of or {}
    child_classes = single_host_child_classes or set()
    hosts = host_classes or set()
    child_has_direct_host: Set[str] = set()
    if child_classes and hosts:
        for rel in out:
            a, b = tuple(rel.pair)
            for child, other in ((a, b), (b, a)):
                if classes.get(child) in child_classes and classes.get(other) in hosts:
                    child_has_direct_host.add(child)

    dropped_gap: List[Tuple[str, str, float]] = []
    dropped_single_host: List[Tuple[str, str]] = []
    dropped_far_terminal: List[Tuple[str, str, int]] = []

    if enable_path_traversal:
        adjacency = LineGraphAdjacency.from_page_line_graph(
            page_graph, junction_to_detection_id=j2d, route_through_inline=route_through_inline)
        explore_depth = max(max_path_depth, terminal_max_depth or max_path_depth)
        all_paths = adjacency.find_symbol_paths(
            max_depth=explore_depth,
            symbol_class_of=symbol_class_of,
            passthrough_symbol_classes=passthrough_symbol_classes,
            terminal_ids=terminal_ids,
            max_passthrough_hops=max_passthrough_hops,
        )

        # Depth gating, applied per edge class rather than globally. A boundary path is
        # inherently longer than an on-sheet one — the pipe crosses the sheet from equipment out
        # to a doorway at the border — and the hop cap of 8 was inherited for symbol<->symbol
        # relations. Measured on PX-2368-0180004-001: all 15 pentagons DO reach a real symbol and
        # none are broken, but they need 2-14 hops (median ~9), so a cap of 8 silently discarded
        # 8 of them. Explore to the larger depth, then keep each path only if it satisfies the
        # cap for its own class.
        def _class_ok(p) -> bool:
            boundary = (p.source_detection_id in terminal) != (p.target_detection_id in terminal)
            cap = (terminal_max_depth or max_path_depth) if boundary else max_path_depth
            return p.n_segments <= cap

        gated = [p for p in all_paths if _class_ok(p)]

        # A doorway is the end of ONE pipe run, so it connects to the symbol that run actually
        # reaches — not to every symbol within the hop cap. Without this, raising the cap enough
        # to reach the far pentagons inflated boundary edges from 7 to 30 (15 ports averaging 2
        # targets each) and to 52 at depth 20. Keeping only the shortest path per terminal is the
        # same principle as the single-host rule for child devices, and it is threshold-free.
        if terminal:
            # The "nearest" contest counts only paths to a REAL SYMBOL. A doorway's meaningful
            # connection is to something on the sheet, never to another doorway — and port<->port
            # is suppressed at emit anyway, so letting one win the contest would starve the port
            # of its real edge entirely. Measured on PX-2368-0180004-001: 7 of 15 pentagons had
            # their shortest path land on another pentagon (4-7 hops, e.g. port001<->port002,
            # port003<->port009) because the border-column structure links them. Counting those
            # left only 7 of 15 doorways connected; excluding them connects all 15.
            def _other(p, nid):
                return p.target_detection_id if p.source_detection_id == nid else p.source_detection_id

            best_hops: Dict[str, int] = {}
            for p in gated:
                for nid in (p.source_detection_id, p.target_detection_id):
                    if nid not in terminal:
                        continue
                    if _other(p, nid) in terminal:
                        continue      # port<->port never sets the bar
                    cur = best_hops.get(nid)
                    if cur is None or p.n_segments < cur:
                        best_hops[nid] = p.n_segments

            kept = []
            for p in gated:
                ok = True
                for nid in (p.source_detection_id, p.target_detection_id):
                    if nid not in terminal or _other(p, nid) in terminal:
                        continue
                    bar = best_hops.get(nid)
                    if bar is not None and p.n_segments > bar:
                        ok = False
                        break
                if ok:
                    kept.append(p)
                else:
                    dropped_far_terminal.append(
                        (p.source_detection_id, p.target_detection_id, p.n_segments))
            gated = kept

        for path in gated:
            src, tgt = path.source_detection_id, path.target_detection_id
            key = tuple(sorted((src, tgt)))
            if key in emitted_pairs:
                continue  # pass 0/1 direct evidence wins over traversal for the same pair

            # per-node precedence: a child device with a known direct host does not acquire
            # additional hosts by transitive walk.
            if child_classes and hosts:
                blocked = False
                for child, other in ((src, tgt), (tgt, src)):
                    if (child in child_has_direct_host
                            and classes.get(child) in child_classes
                            and classes.get(other) in hosts):
                        blocked = True
                        break
                if blocked:
                    dropped_single_host.append((src, tgt))
                    continue

            # Distance cap — ON-SHEET pairs only. A boundary edge (one end an off-page
            # connector) is inherently long: the pipe runs from equipment deep in the drawing
            # out to a doorway at the sheet border, so its endpoint gap is naturally thousands
            # of pixels. Applying an on-sheet transitive-link cap to it deletes every real
            # boundary edge — measured: with the cap applied indiscriminately, 13 port edges
            # were dropped and boundary detection read 0. The cap's purpose is to remove
            # spurious transitive links BETWEEN ON-SHEET SYMBOLS; a doorway is a different case.
            is_boundary = (src in terminal) != (tgt in terminal)
            if max_traversal_gap_px is not None and node_bbox and not is_boundary:
                gap = _bbox_gap(node_bbox.get(src), node_bbox.get(tgt))
                if gap is not None and gap > max_traversal_gap_px:
                    dropped_gap.append((src, tgt, gap))
                    continue

            _emit(src, tgt, f"path_through_{path.n_segments}_segments",
                  dominant_line_type(path), compute_path_confidence(path))

    # ── Post-filter — child<->child suppression without a shared host (2026-07-28) ─────────
    # Root cause of 5 of 7 measured on-sheet false positives: a direct or traversed pipe between
    # two child-class devices (instrument/valve/safety_device) with no equipment involved is
    # usually a false positive on real sheets — border/leader-line artifacts and adjacent bubbles
    # picked up as if physically piped together. But a REAL inline valve<->valve run (two valves
    # literally in series on one equipment's line) is common and correctly real — and on
    # PID2Graph, valve|valve is the single strongest scored stratum (F1 0.314 vs 0.126
    # asset<->asset self-test), so a blanket child<->child ban would be a regression, not a fix.
    # The qualifier: keep a child<->child edge only when BOTH ends have (already-emitted) direct
    # evidence of the SAME host equipment — i.e. they are both known to sit on that one
    # equipment's inline train, the same fact the single-host rule already tracks. A pair where
    # either end's host is unknown, or the hosts differ, is dropped.
    # Only activates when `single_host_child_classes`/`host_classes` are both supplied (same gate
    # as the single-host rule above) — omitting them preserves prior behavior exactly, including
    # every PID2Graph self-test that doesn't pass these kwargs at all.
    # ── Post-pass — child<->child edges must have IN-SERIES evidence ────────────
    # Root cause of 5 of the 7 measured on-sheet false positives on PX-2368-0180004-001: an edge
    # between two child-class devices (instrument / valve / safety_device) that are not physically
    # connected to each other at all.
    #
    # The discriminator is the SOURCE PASS, not the host relationship. An earlier version of this
    # rule keyed on whether the two children shared a host equipment, and that is measurably the
    # wrong test: `LSHL-0100<->TSH-0100` and `PSHL-0100<->LSL-0100B` are two instruments hanging off
    # the SAME vessel by separate stems, with no pipe between them — they share a host and are still
    # false. Meanwhile two valves genuinely in series on one line may sit on different hosts.
    #
    # What actually distinguishes them is whether a single pipe run physically passes through both.
    # That is exactly what Pass 0 (inline-chain, sourced from `segment.passes_through_symbols`)
    # means. So: a child<->child edge survives only with Pass 0 evidence. Pass 1 (endpoint
    # resolution — the mechanism that merges two adjacent bubbles' stems into one segment) and
    # Pass 2 (transitive traversal along a shared header) are precisely the two failure modes.
    #
    # This preserves inline valve<->valve runs, which are real and are the strongest scored stratum
    # on PID2Graph (F1 0.314 vs 0.126 asset<->asset), because those come from Pass 0.
    # Only activates when `single_host_child_classes` is supplied, so any caller that omits it —
    # including every PID2Graph self-test — is untouched.
    dropped_child_child: List[Tuple[str, str]] = []
    if child_classes:
        kept: List[TopologyRelation] = []
        for rel in out:
            a, b = tuple(rel.pair)
            both_children = (classes.get(a) in child_classes
                             and classes.get(b) in child_classes)
            if both_children and not rel.source.startswith("passes_through_"):
                dropped_child_child.append((a, b))
                continue
            kept.append(rel)
        out = kept

    if stats_out is not None:
        # Explicit out-param rather than module state: no silent truncation — a pipeline that
        # drops candidates must be able to say exactly what it dropped. Drop reasons are reported
        # separately so the precision worklist can carry them as their own strata and we can
        # check none of these rules is killing real edges.
        stats_out["suppressed_terminal_pairs"] = list(suppressed_terminal_pairs)
        stats_out["dropped_distance_cap"] = list(dropped_gap)
        stats_out["dropped_single_host"] = list(dropped_single_host)
        stats_out["dropped_far_terminal"] = list(dropped_far_terminal)
        stats_out["dropped_child_child_no_shared_host"] = list(dropped_child_child)

    return out
