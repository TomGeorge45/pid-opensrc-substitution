"""Two pure bbox-geometry functions, extracted from pnid-intelligence-agent's
`pnid_agent/shared/isa_rules.py` (that module also has a large tag-TEXT-parsing chain via
`pnid_agent.grammars`, which is why we didn't vendor the whole file for the PID2Graph work
in Part A — no tag text there to parse). These two functions are geometry-only, no text
dependency at all, and `vector_graph.py` only imports these two names — so they're vendored
standalone here rather than pulling in the ~650-line grammar parser for two functions that
never touch it.
"""
from __future__ import annotations

from typing import Optional


def _percentile(sorted_vals: list, pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def diagram_area_bbox(
    symbol_bboxes: list,
    page_size: Optional[tuple] = None,
    *,
    margin_frac: float = 0.03,
    min_symbols: int = 8,
    low_pct: float = 0.0,
    high_pct: float = 100.0,
) -> Optional[list]:
    """Estimate the drawing's working area from where the real symbols are — the bbox of
    the symbol cloud, padded. Returns None when there are too few symbols to estimate
    reliably (don't clip a sparse page)."""
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in symbol_bboxes if b and len(b) >= 4]
    if len(centers) < min_symbols:
        return None
    xs = sorted(c[0] for c in centers)
    ys = sorted(c[1] for c in centers)
    x0, x1 = _percentile(xs, low_pct), _percentile(xs, high_pct)
    y0, y1 = _percentile(ys, low_pct), _percentile(ys, high_pct)
    if page_size and len(page_size) >= 2:
        margin = margin_frac * max(page_size[0], page_size[1])
    else:
        margin = margin_frac * max(x1 - x0, y1 - y0)
    return [x0 - margin, y0 - margin, x1 + margin, y1 + margin]


def bbox_center_outside(bbox: Optional[list], area: Optional[list]) -> bool:
    """True when `bbox`'s center lies outside `area` ([x0,y0,x1,y1])."""
    if not bbox or len(bbox) < 4 or not area:
        return False
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return not (area[0] <= cx <= area[2] and area[1] <= cy <= area[3])
