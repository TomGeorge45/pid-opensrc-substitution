"""Ported verbatim (logic-for-logic) from pnid-intelligence-agent Stage 11 driver's
``_build_junction_to_detection_map``. After Stage 6 masks symbol bboxes, pipes are severed
at the mask edge — the junction that forms right there IS the symbol's connection point,
so it gets promoted to a symbol node for graph_construction's passes 1/2.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from line_tracing.models import Junction

BBox = Tuple[int, int, int, int]


def build_junction_to_detection_map(
    junctions: List[Junction],
    symbol_bboxes: Sequence[BBox],
    symbol_det_ids: Sequence[str],
    *,
    threshold_px: int = 27,
) -> Dict[str, str]:
    if not symbol_bboxes:
        return {}
    result: Dict[str, str] = {}
    for junc in junctions:
        jx, jy = junc.position
        best_det: Optional[str] = None
        best_dist2 = float("inf")
        for bbox, det_id in zip(symbol_bboxes, symbol_det_ids):
            x0, y0, x1, y1 = bbox
            if not (x0 - threshold_px <= jx <= x1 + threshold_px
                    and y0 - threshold_px <= jy <= y1 + threshold_px):
                continue
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            d2 = (cx - jx) ** 2 + (cy - jy) ** 2
            if d2 < best_dist2:
                best_dist2 = d2
                best_det = det_id
        if best_det:
            result[junc.junction_id] = best_det
    return result
