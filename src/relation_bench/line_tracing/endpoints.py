"""Resolve a polyline endpoint position to a junction, symbol, or loose_end.

Priority:
  1. If the position is within ``junction_snap_radius_px`` of a junction
     → kind=junction, ref=closest junction_id.
  2. Else if the position is inside (or within ``symbol_snap_radius_px``
     of) a symbol bbox → kind=symbol, ref=detection_id of the symbol
     whose CENTER is closest (handles overlapping bbox inflations
     deterministically).
  3. Else → kind=loose_end.

Junction snap radius is small (5-12 px — junctions are clustered already)
but symbol snap radius is generous (20-30 px — pipe endpoints often land
short of the symbol's masked bbox after skeletonization).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .models import Endpoint


BBox = Tuple[int, int, int, int]


def build_junction_grid(
    junction_positions: List[Tuple[int, int]],
    junction_ids: List[str],
    *,
    cell_size_px: int = 5,
) -> Dict[Tuple[int, int], List[Tuple[int, int, str]]]:
    """Build a sparse grid of (cell_x, cell_y) → [(x, y, junction_id), ...].

    The cell size acts as the snap radius bin — a cell can have multiple
    junctions; a snap lookup must check the cell + the 8 surrounding cells.
    """
    if len(junction_positions) != len(junction_ids):
        raise ValueError("junction_positions and junction_ids must be same length")
    grid: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
    for (x, y), jid in zip(junction_positions, junction_ids):
        key = (x // cell_size_px, y // cell_size_px)
        grid.setdefault(key, []).append((x, y, jid))
    return grid


def resolve_endpoint(
    position: Tuple[int, int],
    junction_grid: Dict[Tuple[int, int], List[Tuple[int, int, str]]],
    symbol_bboxes: Sequence[BBox],
    detection_ids: Sequence[str],
    *,
    junction_snap_radius_px: Optional[int] = None,
    symbol_snap_radius_px: Optional[int] = None,
    snap_radius_px: Optional[int] = None,
    cell_size_px: int = 5,
) -> Endpoint:
    """Return the :class:`Endpoint` for this polyline endpoint position.

    Backwards-compat:
      - If ``snap_radius_px`` is provided (legacy V1 callers), it's used
        as BOTH junction and symbol radius unless one or the other is
        explicitly set.
      - New callers should pass ``junction_snap_radius_px`` +
        ``symbol_snap_radius_px`` directly.
      - Defaults: junction=12, symbol=25 (both larger than the prior
        V1 default of 5 — the prior default was tuned conservatively for
        a single-radius world).
    """
    j_radius = (
        junction_snap_radius_px
        if junction_snap_radius_px is not None
        else (snap_radius_px if snap_radius_px is not None else 12)
    )
    s_radius = (
        symbol_snap_radius_px
        if symbol_snap_radius_px is not None
        else (snap_radius_px if snap_radius_px is not None else 25)
    )

    ex, ey = int(position[0]), int(position[1])

    # 1. Snap to nearest junction within junction snap radius.
    best_id = _nearest_junction(
        ex, ey, junction_grid, j_radius, cell_size_px=cell_size_px,
    )
    if best_id is not None:
        return Endpoint(kind="junction", ref=best_id, position=(ex, ey))

    # 2. Inside / near a symbol bbox? When multiple bboxes' inflations
    # contain the point, pick the one whose CENTER is closest.
    best_det = _nearest_symbol(ex, ey, symbol_bboxes, detection_ids, s_radius)
    if best_det is not None:
        return Endpoint(kind="symbol", ref=best_det, position=(ex, ey))

    # 3. Loose end.
    return Endpoint(kind="loose_end", ref=None, position=(ex, ey))


def _nearest_junction(
    ex: int,
    ey: int,
    junction_grid: Dict[Tuple[int, int], List[Tuple[int, int, str]]],
    radius: int,
    *,
    cell_size_px: int,
) -> Optional[str]:
    """Scan the 3x3 neighborhood of cells around (ex, ey); return the
    closest junction_id within `radius`, else None."""
    cell_x = ex // cell_size_px
    cell_y = ey // cell_size_px
    best_id: Optional[str] = None
    best_d2 = (radius + 1) ** 2
    # Expand the neighborhood when radius is larger than cell_size so
    # all candidate cells get checked.
    cells_each_side = max(1, (radius // cell_size_px) + 1)
    for dy in range(-cells_each_side, cells_each_side + 1):
        for dx in range(-cells_each_side, cells_each_side + 1):
            key = (cell_x + dx, cell_y + dy)
            if key not in junction_grid:
                continue
            for jx, jy, jid in junction_grid[key]:
                d2 = (jx - ex) ** 2 + (jy - ey) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = jid
    return best_id


def _nearest_symbol(
    ex: int,
    ey: int,
    symbol_bboxes: Sequence[BBox],
    detection_ids: Sequence[str],
    radius: int,
) -> Optional[str]:
    """Find every symbol whose inflated bbox contains (ex, ey); return the
    one whose center is closest. Returns None when no symbol matches."""
    best_det: Optional[str] = None
    best_d2 = float("inf")
    for bbox, det_id in zip(symbol_bboxes, detection_ids):
        x0, y0, x1, y1 = bbox
        if not (
            x0 - radius <= ex <= x1 + radius
            and y0 - radius <= ey <= y1 + radius
        ):
            continue
        # Inflation contains the endpoint — compute center distance.
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        d2 = (cx - ex) ** 2 + (cy - ey) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_det = det_id
    return best_det
