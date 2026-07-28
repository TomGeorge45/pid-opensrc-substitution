"""Graph traversal over the Stage 6 line graph.

Stage 6 produces segments + junctions. Most real P&ID pipes run through
multiple junctions before terminating at a symbol; the direct-segment
relation extraction misses every multi-segment path. This module walks
the graph to discover symbol-to-symbol paths via BFS, bounded by
``max_depth``.

Output: one :class:`TraversalPath` per (source_detection_id,
target_detection_id) pair — deduped to the SHORTEST path when multiple
exist. The driver feeds each into the ontology relation lookup and
emits one :class:`BundleRelation` per path.

Self-loops (both endpoints reference the same symbol) are excluded by
construction. Paths through OTHER symbols are also excluded — those
become separate, shorter relations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from line_tracing.models import PageLineGraph, Segment


@dataclass(frozen=True)
class TraversalPath:
    """One symbol-to-symbol walk through the Stage 6 line graph."""
    source_detection_id: str
    target_detection_id: str
    segment_ids: Tuple[str, ...]            # ordered start→end
    junction_ids_traversed: Tuple[str, ...]  # the junctions in the middle
    line_types: Tuple[str, ...]             # one per segment
    avg_segment_confidence: float           # baseline before penalties
    n_ambiguous_junctions: int              # cross-vs-jumper count along path

    @property
    def n_segments(self) -> int:
        return len(self.segment_ids)


# Node = junction_id (str) OR ("symbol", detection_id).
_Node = Tuple[str, str]  # ("junction", junction_id) or ("symbol", detection_id)


def _node_for_endpoint(endpoint) -> _Node | None:
    """Map a Stage 6 Endpoint to a graph node key. Returns None for loose_end."""
    if endpoint.kind == "junction":
        return ("junction", str(endpoint.ref)) if endpoint.ref else None
    if endpoint.kind == "symbol":
        return ("symbol", str(endpoint.ref)) if endpoint.ref else None
    return None  # loose_end


@dataclass
class _Edge:
    """Adjacency-list entry: traverse `segment` to reach `other_node`."""
    other_node: _Node
    segment: Segment


class LineGraphAdjacency:
    """In-memory adjacency for one page's line graph.

    Nodes:
      - ("junction", junction_id) for every Stage 6 junction.
      - ("symbol", detection_id)  for every symbol-anchored endpoint.
    Edges:
      - One per Segment, connecting its endpoint_a node to its endpoint_b node.
        Segments with a loose_end endpoint contribute a half-edge that's
        unreachable from the symbol set; we drop them entirely.

    Built once per page; queried with :meth:`find_symbol_paths`.
    """

    def __init__(
        self,
        adjacency: Dict[_Node, List[_Edge]],
        junction_ambiguity: Dict[str, bool],
    ) -> None:
        self._adjacency = adjacency
        self._junction_ambiguity = junction_ambiguity

    @classmethod
    def from_page_line_graph(
        cls,
        page_graph: PageLineGraph,
        *,
        junction_to_detection_id: Optional[Dict[str, str]] = None,
        route_through_inline: bool = False,
    ) -> "LineGraphAdjacency":
        """Build adjacency from the line graph.

        ``junction_to_detection_id`` maps junction_id → detection_id for
        junctions that sit at the boundary of a symbol bbox (produced by
        ``_build_junction_to_detection_map`` in the graph-construct driver).
        When provided, those junctions are promoted to ``("symbol", det_id)``
        nodes so BFS starts from — and terminates at — them, giving the path
        traversal full visibility into symbol connections.
        """
        j2d = junction_to_detection_id or {}

        def _upgraded_node(endpoint) -> "_Node | None":
            if endpoint.kind == "junction" and endpoint.ref:
                det_id = j2d.get(str(endpoint.ref))
                if det_id:
                    return ("symbol", det_id)
                return ("junction", str(endpoint.ref))
            if endpoint.kind == "symbol" and endpoint.ref:
                return ("symbol", str(endpoint.ref))
            return None  # loose_end

        adjacency: Dict[_Node, List[_Edge]] = {}
        for segment in page_graph.segments:
            a = _upgraded_node(segment.endpoint_a)
            b = _upgraded_node(segment.endpoint_b)

            # Inline symbols become ROUTABLE NODES along the segment, not just an annotation.
            # Populating `passes_through_symbols` alone changed nothing (measured: 295 segments
            # annotated, 327 links, zero new relations) because Pass 0 only chains symbols that
            # sit on ONE segment, while an inline valve's neighbours are reached through other
            # segments via junctions — and the valve was never a node in this adjacency at all,
            # since nodes were only ever created from segment ENDPOINTS. So a pipe running
            # straight through a valve produced no way to route to it.
            # Splitting the segment into a chain [a] - sym1 - ... - symN - [b] makes the valve a
            # real waypoint, which is what lets BFS reach it and continue past it.
            mids = ([("symbol", str(s)) for s in (segment.passes_through_symbols or [])]
                    if route_through_inline else [])

            if not mids:
                # EXACT original behaviour. Gated behind `route_through_inline` rather than on
                # "does this segment have inline symbols", because it turns out the tracer DOES
                # populate `passes_through_symbols` under the legacy config — 6,235 of 9,784
                # segments, with 191 of the 237 legacy relations coming from Pass 0. An earlier
                # measurement of "0 of 8,105" was taken on the v2 graph, whose extents are tight
                # enough that almost nothing passes through them; it was never globally dead.
                # So routing through inline symbols must be opt-in or it silently rewrites the
                # legacy result (measured: 78/52 -> 237/106).
                if a is None or b is None:
                    continue  # loose-end segments aren't traversable.
                if a == b:
                    continue  # self-loop on the same node — skip (handled separately).
                adjacency.setdefault(a, []).append(_Edge(other_node=b, segment=segment))
                adjacency.setdefault(b, []).append(_Edge(other_node=a, segment=segment))
                continue

            # Inline symbols become ROUTABLE WAYPOINTS along the segment. A loose end at one
            # side still leaves them reachable from the other, so only the None is dropped.
            chain: List["_Node | None"] = [a] + mids + [b]
            usable = [n for n in chain if n is not None]
            for u, v in zip(usable, usable[1:]):
                if u == v:
                    continue
                adjacency.setdefault(u, []).append(_Edge(other_node=v, segment=segment))
                adjacency.setdefault(v, []).append(_Edge(other_node=u, segment=segment))
        junction_ambiguity = {
            j.junction_id: bool(j.ambiguous) for j in page_graph.junctions
        }
        return cls(adjacency, junction_ambiguity)

    @property
    def nodes(self) -> List[_Node]:
        return list(self._adjacency.keys())

    def find_symbol_paths(
        self, *, max_depth: int = 8,
        symbol_class_of: Optional[Dict[str, str]] = None,
        passthrough_symbol_classes: Optional[Set[str]] = None,
        terminal_ids: Optional[Set[str]] = None,
        max_passthrough_hops: Optional[int] = None,
    ) -> List[TraversalPath]:
        """BFS from each symbol-anchored node; return one path per unique
        (source_detection_id, target_detection_id) pair (shortest wins).

        ``symbol_class_of``/``passthrough_symbol_classes`` (the process-backbone pass,
        Benchmark_Gaps_Register.md gap #14): without these, BFS stops dead at the FIRST
        symbol it reaches from `source` (see ``_bfs_from`` — "don't traverse through it").
        That's correct for direct/short relations but means a real asset<->asset backbone
        that physically runs through several inline valves/instruments (vessel -> valve ->
        valve -> vessel, no direct short segment) is never discovered — this is the
        asset<->asset ceiling documented against R1+R2a (F1 0.126 vs 0.314 for adjacent
        valve|valve pairs). When a symbol's class is in ``passthrough_symbol_classes``, BFS
        BOTH records it as a direct target (unchanged behavior, preserves existing
        valve-adjacent strata) AND continues walking through it toward the next symbol —
        additive, not a replacement of the existing short-hop discovery.

        ``terminal_ids`` (Pipeline 3 v2): node ids a walk may END at but must NEVER pass
        THROUGH, whatever their class says — every off-page connector port. This exists
        because the backbone pass's worst measured failure on a real drawing was walking
        BETWEEN two unrelated border-column doorways: 4 of the 5 edges it added on
        PX-2368-0180004-001 were pairs of vertically adjacent entries in the same connector
        column (`MBF-0623`/`HBG-0905` 228px apart at identical x=5290, `MBF-0500`/`PBM-0450`
        138px, `PBA-0501`/`PBA-0903` 406px at identical x~838). The 762-sheet corpus check
        never caught this because PID2Graph sheets have no border connector columns at all.
        Terminality removes the failure by construction rather than by threshold tuning.

        ``max_passthrough_hops`` (Pipeline 3 v2): cap on how many pass-through symbols a
        single walk may traverse. The corpus check validated exactly ONE case — walking
        through a single inline fitting (84.6% of valve hops and 84.8% of instrumentation hops
        were already real GT edges) — but the uncapped implementation also produces longer
        chains that were never validated, and measured false-positive growth came in at 54%
        against the ~15% the corpus check predicted. Setting this to 1 restricts the pass to
        the case that actually has evidence behind it. ``None`` = uncapped (legacy behavior).

        Both parameters are additive: omit them and behavior is byte-identical to before, so
        every previously-recorded number remains reproducible (the Phase 0 regression gate).
        """
        symbol_nodes = [n for n in self._adjacency if n[0] == "symbol"]
        # Use unordered pair (src, tgt) for dedup so we don't emit both
        # (A→B) and (B→A).
        seen_pairs: Set[Tuple[str, str]] = set()
        paths: List[TraversalPath] = []

        for source in symbol_nodes:
            for path in self._bfs_from(
                source, max_depth=max_depth,
                symbol_class_of=symbol_class_of,
                passthrough_symbol_classes=passthrough_symbol_classes,
                terminal_ids=terminal_ids,
                max_passthrough_hops=max_passthrough_hops,
            ):
                src_det = source[1]
                tgt_det = path.target_detection_id
                if src_det == tgt_det:
                    continue
                pair_key = tuple(sorted((src_det, tgt_det)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                paths.append(path)
        # Stable order by (source_detection_id, target_detection_id) for
        # deterministic relation_id allocation in the driver.
        paths.sort(key=lambda p: (p.source_detection_id, p.target_detection_id))
        return paths

    def _bfs_from(
        self, source: _Node, *, max_depth: int,
        symbol_class_of: Optional[Dict[str, str]] = None,
        passthrough_symbol_classes: Optional[Set[str]] = None,
        terminal_ids: Optional[Set[str]] = None,
        max_passthrough_hops: Optional[int] = None,
    ) -> List[TraversalPath]:
        """BFS from `source`. Each visited node records its parent edge so
        we can reconstruct the path when we hit another symbol. By default, stops
        traversal AT each symbol node (don't walk through it) — UNLESS that symbol's
        class is in ``passthrough_symbol_classes`` (the backbone pass, gap #14), in which
        case the path is still recorded but the walk also continues through it.

        Two v2 guards (see ``find_symbol_paths`` for the measurements behind them):
        a node in ``terminal_ids`` is never walked through even if its class is
        pass-through eligible, and a walk may traverse at most
        ``max_passthrough_hops`` pass-through symbols.
        """
        # parent[node] = (prev_node, edge_taken)
        parent: Dict[_Node, Tuple[_Node, _Edge] | None] = {source: None}
        depth: Dict[_Node, int] = {source: 0}
        # How many pass-through SYMBOLS the walk crossed to reach each node. Tracked
        # separately from `depth` (which counts segments/junctions too) because the corpus
        # check only ever validated the single-fitting case.
        pt_hops: Dict[_Node, int] = {source: 0}
        found: List[TraversalPath] = []
        passthrough = passthrough_symbol_classes or set()
        class_of = symbol_class_of or {}
        terminal = terminal_ids or set()

        # A source that is itself terminal (a port) may still originate a walk — a pipe
        # legitimately starts at a doorway — so only mid-walk traversal is blocked.
        queue: deque[_Node] = deque([source])
        while queue:
            cur = queue.popleft()
            if depth[cur] >= max_depth:
                continue
            for edge in self._adjacency.get(cur, []):
                nxt = edge.other_node
                if nxt in parent:
                    continue
                parent[nxt] = (cur, edge)
                depth[nxt] = depth[cur] + 1
                pt_hops[nxt] = pt_hops[cur]

                if nxt[0] == "symbol" and nxt != source:
                    # Reached another symbol — always record the direct path.
                    found.append(self._build_path(source, nxt, parent))
                    if nxt[1] in terminal:
                        # Off-page connector port: a pipe ENDS here. Never walk through it —
                        # this is what stops two unrelated border-column doorways being
                        # joined into a false edge.
                        continue
                    if class_of.get(nxt[1]) in passthrough:
                        # Backbone pass: an inline pass-through class (valve/instrument/
                        # fitting) -- keep walking toward the next symbol, in ADDITION to
                        # recording it as a direct target above. Bounded by
                        # max_passthrough_hops so only the corpus-validated single-fitting
                        # case fires when the cap is 1.
                        hops = pt_hops[cur] + 1
                        if max_passthrough_hops is not None and hops > max_passthrough_hops:
                            continue
                        pt_hops[nxt] = hops
                        queue.append(nxt)
                    continue
                # It's a junction — keep walking.
                queue.append(nxt)
        return found

    def _build_path(
        self,
        source: _Node,
        target: _Node,
        parent: Dict[_Node, Tuple[_Node, _Edge] | None],
    ) -> TraversalPath:
        """Reconstruct the ordered list of segments + junctions from `source`
        to `target` by walking back through `parent`."""
        # Walk from target back to source, collecting edges.
        edges_reversed: List[_Edge] = []
        junctions_reversed: List[str] = []
        cur: _Node | None = target
        while cur is not None and cur != source:
            p = parent[cur]
            if p is None:
                break
            prev_node, edge = p
            edges_reversed.append(edge)
            # If prev_node is a junction it sits in the middle of the path.
            if prev_node[0] == "junction" and prev_node != source:
                junctions_reversed.append(prev_node[1])
            cur = prev_node
        edges = list(reversed(edges_reversed))
        junctions = list(reversed(junctions_reversed))

        segment_ids = tuple(e.segment.segment_id for e in edges)
        line_types = tuple(str(e.segment.line_type) for e in edges)
        confidences = [float(e.segment.confidence) for e in edges]
        avg = sum(confidences) / len(confidences) if confidences else 0.0
        n_ambiguous = sum(
            1 for jid in junctions if self._junction_ambiguity.get(jid, False)
        )
        return TraversalPath(
            source_detection_id=source[1],
            target_detection_id=target[1],
            segment_ids=segment_ids,
            junction_ids_traversed=tuple(junctions),
            line_types=line_types,
            avg_segment_confidence=avg,
            n_ambiguous_junctions=n_ambiguous,
        )
