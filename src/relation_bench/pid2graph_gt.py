"""PID2Graph ground-truth loading + contraction for the relation-stage benchmark.

WHY CONTRACTION EXISTS (Benchmark_Gaps_Register.md, gap #4): PID2Graph's graphml edges are
LINE-GRAPH edges — they route through `connector` and `crossing` nodes (one per drawn
point/junction along a pipe), e.g. `general1 — connector159 — connector211 — valve210`.
A relation pipeline outputs SYMBOL-to-SYMBOL edges and can never emit an edge whose
endpoint is "crossing47", so scoring raw graphml edges would structurally zero-out every
pipeline regardless of quality (this exact mistake produced the all-zero relation scores
of 2026-07-16, fixed then by the same idea). Contraction collapses every chain of
pass-through nodes into direct symbol↔symbol edges — the shape all three benchmark arms
actually output. This mirrors intelligence-agent Stage 6's own degree-2 contraction, so
it is a principled canonical form, not a benchmark convenience.

FROZEN RULES (documented per the register; do not change silently once arms have run):
  - SYMBOL classes (contraction endpoints): valve, instrumentation, tank, pump, general.
    `general` is included: on real sheets it is the equipment/vessel class (PID2Graph's
    own taxonomy), and asset↔asset is the money metric.
  - PASS-THROUGH classes (collapsed): connector, crossing, arrow.
  - DROPPED entirely: background (annotation artifacts, never process content).
  - Contracted edge = two DISTINCT symbol nodes reachable through a path whose interior
    nodes are all pass-through. Implemented via connected components of the pass-through
    subgraph: symbols adjacent to the same pass-through component are pairwise connected;
    plus direct symbol-symbol graphml edges pass through unchanged.
  - UNDIRECTED, deduped (gap #16: graphml source→target direction does not encode process
    flow reliably; extraction-agent emits directed guesses the GT cannot adjudicate).
  - line_type of a contracted edge: 'dashed' if ANY segment on the connecting component's
    incident edges is dashed (dashed = signal/electric per the dataset's convention),
    else 'solid'. Mixed chains are rare; the any-dashed rule is conservative for the
    solid-stratum (process-line) metric.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

SYMBOL_CLASSES = {"valve", "instrumentation", "tank", "pump", "general"}
PASSTHROUGH_CLASSES = {"connector", "crossing", "arrow"}
DROP_CLASSES = {"background"}

_GML_NS = "{http://graphml.graphdrawing.org/xmlns}"


@dataclass
class GTNode:
    node_id: str
    cls: str
    bbox: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass
class SheetGT:
    sheet_id: str
    nodes: Dict[str, GTNode]                          # symbol nodes only
    edges: Dict[FrozenSet[str], str]                  # {a,b} -> line_type
    raw_node_count: int = 0
    raw_edge_count: int = 0

    @property
    def edge_pairs(self) -> Set[FrozenSet[str]]:
        return set(self.edges.keys())

    def stratum(self, pair: FrozenSet[str]) -> str:
        """Endpoint-class stratum, order-independent (e.g. 'general|valve')."""
        classes = sorted(self.nodes[n].cls for n in pair)
        return "|".join(classes)


def parse_graphml(path: Path) -> Tuple[Dict[str, GTNode], List[Tuple[str, str, str]], Dict]:
    """Returns (all nodes incl. pass-through, raw edges as (src, dst, line_type), keymap).

    Handles both key layouts seen in the corpus (d1-d4 vs d5-d8 for bbox) by reading the
    <key> declarations instead of hardcoding ids — the two trees (OPEN100 / Dataset PID)
    differ, and the v3 notebook's parser confirmed schema-by-keymap is the robust route.
    """
    root = ET.parse(path).getroot()
    keymap = {}
    for k in root.iter(f"{_GML_NS}key"):
        keymap[k.get("id")] = (k.get("for"), k.get("attr.name"))

    nodes: Dict[str, GTNode] = {}
    for nd in root.iter(f"{_GML_NS}node"):
        vals = {}
        for d in nd.iter(f"{_GML_NS}data"):
            which = keymap.get(d.get("key"))
            if which is None:
                continue
            _for, name = which
            vals[name] = d.text
        cls = (vals.get("label") or "").strip()
        try:
            bbox = (float(vals["xmin"]), float(vals["ymin"]),
                    float(vals["xmax"]), float(vals["ymax"]))
        except (KeyError, TypeError, ValueError):
            continue  # node without a usable box can't anchor anything
        nodes[nd.get("id")] = GTNode(node_id=nd.get("id"), cls=cls, bbox=bbox)

    edges: List[Tuple[str, str, str]] = []
    for eg in root.iter(f"{_GML_NS}edge"):
        line_type = "solid"
        for d in eg.iter(f"{_GML_NS}data"):
            which = keymap.get(d.get("key"))
            if which and which[1] == "edge_label" and d.text:
                line_type = d.text.strip() or "solid"
        edges.append((eg.get("source"), eg.get("target"), line_type))
    return nodes, edges, keymap


def contract(sheet_id: str, nodes: Dict[str, GTNode],
             raw_edges: List[Tuple[str, str, str]]) -> SheetGT:
    """Collapse pass-through chains into direct symbol<->symbol undirected edges."""
    raw_n, raw_e = len(nodes), len(raw_edges)

    keep = {nid: n for nid, n in nodes.items() if n.cls in SYMBOL_CLASSES}
    passthrough = {nid for nid, n in nodes.items() if n.cls in PASSTHROUGH_CLASSES}
    # anything in DROP_CLASSES or unknown classes simply doesn't exist for the benchmark

    adj: Dict[str, Set[str]] = defaultdict(set)
    edge_type: Dict[FrozenSet[str], str] = {}
    for src, dst, lt in raw_edges:
        if src not in nodes or dst not in nodes or src == dst:
            continue
        if nodes[src].cls in DROP_CLASSES or nodes[dst].cls in DROP_CLASSES:
            continue
        adj[src].add(dst)
        adj[dst].add(src)
        key = frozenset((src, dst))
        # any-dashed wins at the raw level too
        if edge_type.get(key) != "dashed":
            edge_type[key] = lt if lt == "dashed" else edge_type.get(key, lt)

    # union-find over the pass-through subgraph
    parent: Dict[str, str] = {nid: nid for nid in passthrough}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for src, dst, _lt in raw_edges:
        if src in passthrough and dst in passthrough:
            union(src, dst)

    # component -> (touching symbols, any-dashed flag over ALL edges incident to the comp)
    comp_symbols: Dict[str, Set[str]] = defaultdict(set)
    comp_dashed: Dict[str, bool] = defaultdict(bool)
    for src, dst, lt in raw_edges:
        if src not in nodes or dst not in nodes:
            continue
        for a, b in ((src, dst), (dst, src)):
            if a in passthrough:
                root_ = find(a)
                if lt == "dashed":
                    comp_dashed[root_] = True
                if b in keep:
                    comp_symbols[root_].add(b)

    contracted: Dict[FrozenSet[str], str] = {}

    # 1) direct symbol-symbol raw edges survive as-is
    for key, lt in edge_type.items():
        a, b = tuple(key)
        if a in keep and b in keep:
            if contracted.get(key) != "dashed":
                contracted[key] = lt if lt == "dashed" else contracted.get(key, lt)

    # 2) symbols sharing a pass-through component become pairwise connected
    for root_, syms in comp_symbols.items():
        lt = "dashed" if comp_dashed[root_] else "solid"
        syms_l = sorted(syms)
        for i in range(len(syms_l)):
            for j in range(i + 1, len(syms_l)):
                key = frozenset((syms_l[i], syms_l[j]))
                if contracted.get(key) != "dashed":
                    contracted[key] = lt if lt == "dashed" else contracted.get(key, lt)

    return SheetGT(sheet_id=sheet_id, nodes=keep, edges=contracted,
                   raw_node_count=raw_n, raw_edge_count=raw_e)


def load_sheet(graphml_path: Path, sheet_id: str | None = None) -> SheetGT:
    nodes, raw_edges, _ = parse_graphml(graphml_path)
    return contract(sheet_id or graphml_path.stem, nodes, raw_edges)
