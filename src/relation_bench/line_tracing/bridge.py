"""Bridge polylines across masked symbols.

When mask_symbols painted Stage 4 detection bboxes to 0, pipes that
physically pass THROUGH a valve (or any symbol) got chopped. We reconnect
them: if two polyline endpoints lie on opposite sides of the same symbol
bbox AND are within ``max_gap_px`` of the bbox edges, we merge the
polylines and record the symbol's detection_id as "passed through".

The opposite-side test prevents bridging two pipes that happen to touch
the same bbox edge — those aren't physically continuous.

PROD:
  - ``max_gap_px`` is a hardcoded V1 heuristic (30 px). Real value
    depends on drawing DPI and symbol size. Calibrate from a labeled
    corpus once we have one. Track
    ``pid_stage_06_bridge_attempts_total{outcome}`` for false-bridge rate.
  - See PRODUCTION_TODO.md → "Stage 6 — bridge_max_gap_px tuning".
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)
Polyline = List[Tuple[int, int]]


def bridge_across_symbols(
    polylines: Sequence[Polyline],
    symbol_bboxes: Sequence[BBox],
    detection_ids: Sequence[str],
    *,
    max_gap_px: int = 60,
) -> List[Tuple[Polyline, List[str]]]:
    """Merge polylines whose endpoints flank a symbol bbox.

    Returns a list of ``(polyline, passed_through_detection_ids)``.
    Polylines that weren't bridged appear as singletons with an empty
    ``passed_through`` list.
    """
    if len(symbol_bboxes) != len(detection_ids):
        raise ValueError("symbol_bboxes and detection_ids must be same length")

    # Represent each polyline by its endpoints + whether it's still
    # available for merging.
    state: List[Tuple[Polyline, List[str], bool]] = [
        (list(p), [], True) for p in polylines
    ]

    changed = True
    while changed:
        changed = False
        for bbox, det_id in zip(symbol_bboxes, detection_ids):
            x0, y0, x1, y1 = bbox
            matches: List[int] = []
            for idx, (poly, _passed, alive) in enumerate(state):
                if not alive:
                    continue
                if _endpoint_flanks_bbox(poly[0], bbox, max_gap_px) or \
                   _endpoint_flanks_bbox(poly[-1], bbox, max_gap_px):
                    matches.append(idx)
            # Need ≥2 polylines flanking this bbox to bridge.
            if len(matches) < 2:
                continue
            # Find a pair on OPPOSITE sides (left/right or top/bottom).
            pair = _find_opposite_side_pair(state, matches, bbox)
            if pair is None:
                continue
            i, j = pair
            poly_i, passed_i, _ = state[i]
            poly_j, passed_j, _ = state[j]
            merged = _merge_polylines(poly_i, poly_j, bbox)
            new_passed = sorted(set(passed_i + passed_j + [det_id]))
            state[i] = (merged, new_passed, True)
            state[j] = ([], [], False)  # consumed
            changed = True
            break  # restart outer loop with updated state

    return [(poly, passed) for poly, passed, alive in state if alive and poly]


def _endpoint_flanks_bbox(
    endpoint: Tuple[int, int],
    bbox: BBox,
    max_gap_px: int,
) -> bool:
    """True iff endpoint lies within max_gap_px of any bbox edge AND outside
    the bbox interior (the polyline ends near the symbol edge, not inside)."""
    ex, ey = endpoint
    x0, y0, x1, y1 = bbox
    if x0 < ex < x1 and y0 < ey < y1:
        return False  # endpoint is INSIDE the bbox — odd, but not a flank
    # Distance to each edge.
    dist_left = abs(ex - x0)
    dist_right = abs(ex - x1)
    dist_top = abs(ey - y0)
    dist_bot = abs(ey - y1)
    # Endpoint flanks the bbox if it's close to one edge AND within the
    # other-axis range of the bbox.
    if (dist_left <= max_gap_px or dist_right <= max_gap_px) and y0 - max_gap_px <= ey <= y1 + max_gap_px:
        return True
    if (dist_top <= max_gap_px or dist_bot <= max_gap_px) and x0 - max_gap_px <= ex <= x1 + max_gap_px:
        return True
    return False


def _which_side(endpoint: Tuple[int, int], bbox: BBox) -> str:
    """Return one of {'left','right','top','bottom'} indicating the side
    of the bbox the endpoint most clearly lies on."""
    ex, ey = endpoint
    x0, y0, x1, y1 = bbox
    candidates = {
        "left": x0 - ex if ex <= x0 else float("inf"),
        "right": ex - x1 if ex >= x1 else float("inf"),
        "top": y0 - ey if ey <= y0 else float("inf"),
        "bottom": ey - y1 if ey >= y1 else float("inf"),
    }
    return min(candidates, key=lambda k: candidates[k])


_OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _find_opposite_side_pair(
    state: Sequence[Tuple[Polyline, List[str], bool]],
    indices: Sequence[int],
    bbox: BBox,
):
    """Return a (i, j) pair from ``indices`` whose flanking endpoints lie on
    OPPOSITE sides of bbox; None if no such pair exists."""
    sides = []  # (index, side, endpoint_pos)
    for idx in indices:
        poly, _, _ = state[idx]
        for ep in (poly[0], poly[-1]):
            s = _which_side(ep, bbox)
            sides.append((idx, s, ep))
    for a_idx, a_side, _ in sides:
        for b_idx, b_side, _ in sides:
            if a_idx == b_idx:
                continue
            if _OPPOSITE.get(a_side) == b_side:
                return (a_idx, b_idx)
    return None


def _merge_polylines(a: Polyline, b: Polyline, bbox: BBox) -> Polyline:
    """Join a + b so the joined sequence runs through the bbox.

    We orient each polyline so its bbox-adjacent endpoint is the join
    point, then concatenate.
    """
    def _bbox_distance(p):
        x, y = p
        x0, y0, x1, y1 = bbox
        # Distance to nearest edge.
        if x0 <= x <= x1:
            return min(abs(y - y0), abs(y - y1))
        if y0 <= y <= y1:
            return min(abs(x - x0), abs(x - x1))
        # Corner case — distance to nearest corner.
        return min(
            abs(x - x0) + abs(y - y0),
            abs(x - x1) + abs(y - y0),
            abs(x - x0) + abs(y - y1),
            abs(x - x1) + abs(y - y1),
        )

    # Orient a so its bbox-near endpoint is at the end.
    a_oriented = a if _bbox_distance(a[-1]) < _bbox_distance(a[0]) else list(reversed(a))
    # Orient b so its bbox-near endpoint is at the start.
    b_oriented = b if _bbox_distance(b[0]) < _bbox_distance(b[-1]) else list(reversed(b))
    return a_oriented + b_oriented
