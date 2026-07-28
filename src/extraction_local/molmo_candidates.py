"""Molmo2 point -> OCR-word pairing (plan §3.1). PURE function, no I/O — easily
unit-testable, and swappable later for whatever tiling/pointing-parse code
actually produces `points_by_class` (Phase 2, GPU).

Candidate dict shape matches `grounded_read.py`'s real `Candidate` dicts exactly
so it slots straight into `_route_b_candidates`'s
`snap_candidates(shape_cands + region_cands + ocr_cands + molmo_cands, symbols, R)`
merge (plan §3.1, step 6):
    {"text": <norm>, "raw": <ocr word text>, "box": (x0,y0,x1,y1),
     "source": "molmo_point", "shape": "circle", "signals": ["molmo_point"]}

Points with NO OCR word within `radius` still contribute a synthetic ~2R square
box to `extra_symbol_boxes` (for `symbols` list extension / adjudication), but
emit NO candidate — a tagless candidate can't help revR and only pollutes
reconcile (plan §3.1 step 5).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

Word = Tuple[str, float, float, float, float]     # text, x0, y0, x1, y1
Point = Tuple[float, float]                        # x, y (page/image pixels)
Candidate = Dict[str, object]
Box = Tuple[float, float, float, float]

_AGENT_DIR = "/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-extraction-agent"
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)
from pnid_pipeline.grammar import alnum, norm_text  # noqa: E402


def _dedup_points(points_by_class: Dict[str, List[Point]], min_dist: float) -> List[Tuple[str, float, float]]:
    """Flatten {class: [(x,y), ...]} into [(cls, x, y), ...], dropping any point
    within `min_dist` of an already-kept point (tile-overlap dupes), keeping the
    FIRST occurrence in iteration order (dict order, then list order)."""
    kept: List[Tuple[str, float, float]] = []
    for cls, pts in points_by_class.items():
        for (x, y) in pts:
            is_dupe = False
            for (_kc, kx, ky) in kept:
                if ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5 < min_dist:
                    is_dupe = True
                    break
            if not is_dupe:
                kept.append((cls, x, y))
    return kept


def molmo_candidates(
    points_by_class: Dict[str, List[Point]],
    ocr_words: List[Word],
    radius: float = 120,
) -> Tuple[List[Candidate], List[Box]]:
    """For each (deduped) Molmo point, find the nearest OCR word within `radius`
    px. A hit emits a candidate dict; a miss still contributes the point's
    synthetic ~2R box to `extra_symbol_boxes` (no candidate). Returns
    (candidates, extra_symbol_boxes)."""
    points = _dedup_points(points_by_class, min_dist=radius / 2.0)

    candidates: List[Candidate] = []
    extra_symbol_boxes: List[Box] = []

    for (_cls, x, y) in points:
        best_word = None
        best_dist = radius
        for w in ocr_words:
            text, x0, y0, x1, y1 = w
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d <= best_dist:
                best_dist = d
                best_word = w

        box = (x - radius, y - radius, x + radius, y + radius)
        extra_symbol_boxes.append(box)

        if best_word is not None:
            text, x0, y0, x1, y1 = best_word
            candidates.append({
                "text": norm_text(text),
                "raw": text,
                "box": (x0, y0, x1, y1),
                "source": "molmo_point",
                "shape": "circle",
                "signals": ["molmo_point"],
            })
        # else: no word within radius -> no candidate (tagless candidate helps no
        # one), but the synthetic box above still extends `symbols` for adjudication.

    return candidates, extra_symbol_boxes
