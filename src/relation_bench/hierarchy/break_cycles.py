"""Cycle-break, ported verbatim (algorithm-for-algorithm) from pnid-extraction-agent's
``pnid_pipeline/hierarchy.py::_break_cycles``. The real function operates on ``Tag``
pydantic objects duck-typed on ``.id``/``.parent_id``/``.text``; here it takes a plain
``HNode`` dataclass since PID2Graph nodes carry no tag text (see R2b module docstring in
``priors.py`` for why the parent-ASSIGNMENT logic itself had to be redesigned — this
cycle-break step is the one piece that ports over unchanged).

One deliberate deviation: the real tie-break for which node in a cycle becomes the root is
"shortest tag text wins" (e.g. 'P-5001' over 'P-5001/P-5002') — there is no text here, so
the tie-break is smallest bbox area (a node fully nested inside another's footprint reads
as the more specific/child one on a P&ID; if areas tie, lowest node_id for determinism).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class HNode:
    node_id: str
    parent_id: Optional[str] = None
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def break_cycles(nodes: List[HNode]) -> int:
    """Ensure the parent graph is a forest. Follow each parent_id chain; when it revisits
    a node, that's a cycle — cut it by making the cycle's SMALLEST-area node a root (mirrors
    the real function's "shortest name is the natural ancestor" logic, substituting bbox
    area for tag-text length since these nodes have no text). Also cuts self-parents and
    parents outside the node set. Returns the number of edges cut."""
    by_id: Dict[str, HNode] = {n.node_id: n for n in nodes}
    removed = 0
    for start in nodes:
        path: List[str] = []
        cur: Optional[HNode] = start
        while cur and cur.parent_id:
            nxt = cur.parent_id
            if nxt == cur.node_id or nxt not in by_id:
                cur.parent_id = None
                removed += 1
                break
            if nxt in path:
                cyc = path[path.index(nxt):]
                root_id = min(cyc, key=lambda i: (by_id[i].area, i))
                by_id[root_id].parent_id = None
                removed += 1
                break
            path.append(cur.node_id)
            cur = by_id.get(nxt)
    return removed
