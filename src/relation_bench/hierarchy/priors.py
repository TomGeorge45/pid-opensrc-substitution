"""R2b — hierarchy geometry pass (Benchmark_Gaps_Register.md gap #11).

Real prod hierarchy (pnid-extraction-agent's ``apply_hierarchy``) assigns parent/level via
an LLM reasoning over tag TEXT (name-nesting prefixes, ISA loop numbers) plus the drawing
image. PID2Graph nodes have neither tag text nor an image crop per node — just a class
label and a bbox (class-agnostic-adjacent, like Gupta: rule 5 in CLAUDE.md). So this is NOT
a port of the real assignment logic — it's the geometric prior the register asked for
(bbox containment + connectivity-nearest-equipment), designed fresh for this benchmark.
Only the cycle-break step (``break_cycles.py``) is a real, verbatim-ported piece.

Two priors, in precedence order (first match wins per child node):
  1. Containment — a CHILD-class node whose bbox sits (almost) entirely inside an
     EQUIPMENT-class node's padded bbox is that equipment's child. Ties broken by
     smallest containing-equipment area (the tightest enclosing box is the most specific
     parent, mirroring how a real engineer would read nested symbols).
  2. Connectivity-nearest — for a child node with no containment match, look at its
     R2a topology neighbors; if it's connected (directly or via one hop) to one or more
     EQUIPMENT-class nodes, assign the nearest one by bbox-center distance.
  Nodes matching neither stay parentless (top-level) — this benchmark does not attempt
  the real pass's "system" grouping node, which needs a sheet-level area/unit code no
  PID2Graph class encodes.

IMPORTANT — there is no hierarchy ground truth on PID2Graph (Benchmark_Gaps_Register.md
Group 2, gap #2, still unanswered). This module can be self-tested for STRUCTURAL
soundness (valid forest, no cycles, plausible assignments) but not scored with P/R/F1
until an annotated fixture exists.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple

from pid2graph_gt import GTNode
from .break_cycles import HNode, break_cycles

EQUIP_CLASSES = {"general", "tank", "pump"}
CHILD_CLASSES = {"valve", "instrumentation"}

# Pipeline 3 v2 — the same two roles expressed in real-drawing Tag.type vocabulary, so
# `build_hierarchy` can run on an EntitySet's resolved symbols as well as on PID2Graph GTNodes.
#
# Why this matters more than it looks: the containment prior has NEVER ONCE FIRED. On the
# OPEN100/0 self-test only 13 of 93 nodes got a parent and every single one came from the
# connectivity-nearest fallback, with zero containment matches. That was read at the time as
# "P&ID symbols rarely nest bboxes on this corpus", but the real cause is now measurable: an
# equipment tag arrives as a ~120x20px NAME PLATE (median over 31 equipment tags on
# PX-2368-0180004-001), and a 20px-tall plate cannot geometrically contain a 42px ISA bubble.
# Half of this module has therefore been dead code. With a real vessel extent, an instrument
# bubble genuinely does sit inside it, so the prior should start producing parents.
# PREDICTED, not measured — the magnitude is unknown until the A/B runs.
V2_EQUIP_TYPES = {"equipment"}
V2_CHILD_TYPES = {"valve", "instrument", "safety_device"}


def _contained(inner: tuple, outer: tuple, *, pad_frac: float = 0.05) -> bool:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    pad_x = pad_frac * max(1.0, ox1 - ox0)
    pad_y = pad_frac * max(1.0, oy1 - oy0)
    return (ix0 >= ox0 - pad_x and iy0 >= oy0 - pad_y
            and ix1 <= ox1 + pad_x and iy1 <= oy1 + pad_y)


def _area(bbox: tuple) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _dist2(a: tuple, b: tuple) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def build_hierarchy(
    nodes: Dict[str, GTNode],
    topology_pairs: Set[FrozenSet[str]],
    *,
    equip_classes: Optional[Set[str]] = None,
    child_classes: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Optional[str]], int]:
    """Returns ({node_id: parent_id or None}, edges_cut) — a valid forest (cycles broken).

    `equip_classes`/`child_classes` default to PID2Graph's class names (unchanged behavior,
    so every existing self-test result is reproducible). Pipeline 3 v2 passes
    `V2_EQUIP_TYPES`/`V2_CHILD_TYPES` instead so the same logic runs over real-drawing
    Tag.type values — see `entities_to_hnodes` for the EntitySet adapter."""
    equips = equip_classes if equip_classes is not None else EQUIP_CLASSES
    children = child_classes if child_classes is not None else CHILD_CLASSES
    equip_ids = [nid for nid, n in nodes.items() if n.cls in equips]
    child_ids = [nid for nid, n in nodes.items() if n.cls in children]

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for pair in topology_pairs:
        if len(pair) != 2:
            continue
        a, b = tuple(pair)
        if a in nodes and b in nodes:
            adjacency[a].add(b)
            adjacency[b].add(a)

    parent: Dict[str, Optional[str]] = {nid: None for nid in nodes}

    # 1) containment prior
    for cid in child_ids:
        child_bbox = nodes[cid].bbox
        candidates = [eid for eid in equip_ids if _contained(child_bbox, nodes[eid].bbox)]
        if candidates:
            parent[cid] = min(candidates, key=lambda eid: _area(nodes[eid].bbox))

    # 2) connectivity-nearest prior (only for children containment didn't place)
    for cid in child_ids:
        if parent[cid] is not None:
            continue
        one_hop = {n for n in adjacency.get(cid, set()) if n in equip_ids}
        two_hop: Set[str] = set()
        for mid in adjacency.get(cid, set()):
            two_hop |= {n for n in adjacency.get(mid, set()) if n in equip_ids}
        neighbor_equip = one_hop | two_hop
        if neighbor_equip:
            cc = nodes[cid].center
            parent[cid] = min(neighbor_equip, key=lambda eid: _dist2(cc, nodes[eid].center))

    hnodes = [HNode(node_id=nid, parent_id=parent[nid], bbox=nodes[nid].bbox) for nid in nodes]
    cut = break_cycles(hnodes)
    return {hn.node_id: hn.parent_id for hn in hnodes}, cut


@dataclass
class _NodeView:
    """Minimal GTNode-shaped view (.bbox, .cls, .center) over a v2 SymbolNode, so
    `build_hierarchy` needs no knowledge of the EntitySet types."""
    bbox: tuple
    cls: str

    @property
    def center(self) -> tuple:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def entities_to_hnodes(entity_set) -> Dict[str, "_NodeView"]:
    """Adapt an `entities.EntitySet`'s resolved symbols into the {id: node} mapping
    `build_hierarchy` expects. Ports are excluded — a doorway is not a containment parent or
    child, it is the sheet boundary. Symbols with no resolved extent are excluded rather than
    falling back to a name plate."""
    out: Dict[str, _NodeView] = {}
    for s in entity_set.symbols:
        if s.extent is None:
            continue
        out[s.id] = _NodeView(bbox=tuple(s.extent), cls=s.type or "other")
    return out


def build_hierarchy_v2(entity_set, topology_pairs: Set[FrozenSet[str]]):
    """Pipeline 3 v2 hierarchy — same priors, real extents, real-drawing type vocabulary."""
    return build_hierarchy(
        entities_to_hnodes(entity_set),
        topology_pairs,
        equip_classes=V2_EQUIP_TYPES,
        child_classes=V2_CHILD_TYPES,
    )
