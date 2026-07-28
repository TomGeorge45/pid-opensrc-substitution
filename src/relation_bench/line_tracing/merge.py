"""Step 7.5 — merge collinear line fragments (dash chains, crossing breaks).

The skeleton vectorizer emits one polyline per CONNECTED ink run, so a dashed
instrument-signal line becomes dozens of tiny segments and a process line
broken by a crossing becomes two. Every such break is two loose ends Stage 11
can't anchor — the dominant source of ``loose_pipe_end`` unresolved items.

This pass joins fragment pairs that are geometrically one line:

  • their endpoints are within ``max_join_gap_px`` of each other,
  • both fragments continue in the same direction across the join
    (within ``max_angle_deg``; relaxed for very short dash ticks where
    integer-pixel quantization makes directions noisy), and
  • the join midpoint is NOT near a detected junction — a T- or X-junction
    is real connectivity, not a fragment break, and merging across it would
    bypass the junction's adjacency in Stage 11 path traversal.

Pure geometry — no I/O, no LLM. Mirrors the iterate-until-fixpoint shape of
``bridge_across_symbols`` (bridge.py), which runs just before this pass.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Point = Tuple[int, int]
PolyItem = Tuple[List[Point], List[str]]   # (polyline, passed_through_detection_ids)

# Direction lookback: use up to this much trailing length to estimate an
# endpoint's direction. RDP-simplified polylines have long edges so the last
# edge alone is usually fine; the lookback stabilizes short jagged tails.
_DIR_LOOKBACK_PX = 14.0
# Fragments shorter than this get a relaxed angle tolerance: a 5px dash tick
# with integer coords can be off by ~20 deg without being a different line.
_SHORT_FRAG_PX = 9.0
_SHORT_FRAG_ANGLE_DEG = 27.0


def _length(poly: Sequence[Point]) -> float:
    return sum(
        math.dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1)
    )


def _end_direction(poly: Sequence[Point], end: int) -> Tuple[float, float]:
    """Unit vector pointing OUT of the polyline at the given end (0=start, 1=end)."""
    pts = list(poly) if end == 1 else list(reversed(poly))
    # Walk back from the tip until we accumulate _DIR_LOOKBACK_PX of length.
    tip = pts[-1]
    acc = 0.0
    anchor = pts[-2]
    for i in range(len(pts) - 2, -1, -1):
        acc += math.dist(pts[i], pts[i + 1])
        anchor = pts[i]
        if acc >= _DIR_LOOKBACK_PX:
            break
    dx, dy = tip[0] - anchor[0], tip[1] - anchor[1]
    n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


def _angle_between(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.degrees(math.acos(dot))


def _grid_key(p: Point, cell: float) -> Tuple[int, int]:
    return (int(p[0] // cell), int(p[1] // cell))


def merge_collinear_fragments(
    items: List[PolyItem],
    junction_positions: Sequence[Point],
    *,
    max_join_gap_px: int = 36,
    max_angle_deg: float = 14.0,
    junction_block_radius_px: int = 12,
) -> List[PolyItem]:
    """Iteratively merge fragment pairs that are one geometric line.

    Returns a new list; input is not mutated. ``passed_through`` lists are
    concatenated on merge (order: a then b).
    """
    if not items:
        return items

    # Junction lookup grid (static across iterations).
    jcell = float(max(junction_block_radius_px, 1))
    jgrid: Dict[Tuple[int, int], List[Point]] = {}
    for jp in junction_positions:
        jgrid.setdefault(_grid_key(jp, jcell), []).append(jp)

    def near_junction(p: Tuple[float, float]) -> bool:
        kx, ky = int(p[0] // jcell), int(p[1] // jcell)
        for gx in (kx - 1, kx, kx + 1):
            for gy in (ky - 1, ky, ky + 1):
                for jp in jgrid.get((gx, gy), ()):
                    if math.hypot(p[0] - jp[0], p[1] - jp[1]) <= junction_block_radius_px:
                        return True
        return False

    work: List[PolyItem] = [(list(poly), list(pt)) for poly, pt in items]
    cell = float(max(max_join_gap_px, 1))

    for _ in range(64):   # fixpoint loop; dash chains collapse in O(log n) rounds
        # Endpoint grid: (item_idx, end) at each cell.
        grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for idx, (poly, _) in enumerate(work):
            for end in (0, 1):
                tip = poly[-1] if end == 1 else poly[0]
                grid.setdefault(_grid_key(tip, cell), []).append((idx, end))

        consumed: set = set()
        merged_any = False
        out: List[PolyItem] = []

        for idx, (poly, passed) in enumerate(work):
            if idx in consumed:
                continue
            best = None   # (gap, jdx, jend, end)
            for end in (0, 1):
                tip = poly[-1] if end == 1 else poly[0]
                out_dir = _end_direction(poly, end)
                kx, ky = _grid_key(tip, cell)
                for gx in (kx - 1, kx, kx + 1):
                    for gy in (ky - 1, ky, ky + 1):
                        for jdx, jend in grid.get((gx, gy), ()):
                            if jdx == idx or jdx in consumed:
                                continue
                            jpoly, _jp = work[jdx]
                            jtip = jpoly[-1] if jend == 1 else jpoly[0]
                            gap = math.dist(tip, jtip)
                            if gap > max_join_gap_px:
                                continue
                            # Angle tolerance, relaxed for short ticks.
                            tol = max_angle_deg
                            if (
                                _length(poly) < _SHORT_FRAG_PX
                                or _length(jpoly) < _SHORT_FRAG_PX
                            ):
                                tol = _SHORT_FRAG_ANGLE_DEG
                            j_out = _end_direction(jpoly, jend)
                            # The two outward directions must be opposite.
                            if _angle_between(out_dir, (-j_out[0], -j_out[1])) > tol:
                                continue
                            # For real gaps, the join vector must follow A's direction.
                            if gap > 2.0:
                                jv = (
                                    (jtip[0] - tip[0]) / gap,
                                    (jtip[1] - tip[1]) / gap,
                                )
                                if _angle_between(out_dir, jv) > tol:
                                    continue
                            # Never merge across a real junction.
                            mid = ((tip[0] + jtip[0]) / 2.0, (tip[1] + jtip[1]) / 2.0)
                            if near_junction(mid):
                                continue
                            if best is None or gap < best[0]:
                                best = (gap, jdx, jend, end)
            if best is None:
                out.append((poly, passed))
                continue

            _, jdx, jend, end = best
            jpoly, jpassed = work[jdx]
            # Orient: a's merge end last, b's merge end first.
            a = poly if end == 1 else list(reversed(poly))
            b = jpoly if jend == 0 else list(reversed(jpoly))
            if a[-1] == b[0]:
                b = b[1:]
            out.append((a + b, passed + jpassed))
            consumed.add(idx)
            consumed.add(jdx)
            merged_any = True

        work = out
        if not merged_any:
            break

    return work
