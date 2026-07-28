"""Molmo2 point -> synthetic OCR-token rows for the `ocr_reasoning` pipeline path
(Extraction_Agent_Local_Plan.md §3.A, Phase 1b item 6). PURE function, no I/O.

`ocr_reasoning_extract` (pnid_pipeline/ocr_reasoning.py) never crops per-symbol —
it reasons over ONE flat list of `(id, text, cx_norm, cy_norm)` OCR tokens plus one
overview image (see `_prompt`/`_recovery_prompt`, and the `idx_words` construction
at the top of `ocr_reasoning_extract`: `cx = (x0+x1)/2/W, cy = (y0+y1)/2/H`). Molmo2's
role in this path is to catch symbols OCR's text detector missed ENTIRELY (no token
at all near that location) by injecting extra synthetic rows into that same list:

  - a point with a real OCR word already within `near_radius` is SKIPPED — OCR
    already covers this location; an extra synthetic token there would just be a
    noisy duplicate/double-vote.
  - a point with no word within `near_radius` but one within the larger
    `pair_radius` borrows that word's text (it's probably the tag OCR half-read or
    mis-boxed near the symbol).
  - a point with nothing nearby at all still gets emitted, with placeholder text
    `"?"` — a real symbol location with no legible text nearby; the reasoning LLM
    decides whether to keep it as a tag (matches the pipeline's "identify what's
    real" philosophy — this function never fabricates a tag value).

Returned rows are shaped EXACTLY like `idx_words`: `(id, text, cx_norm, cy_norm)`.
`id`s start at `start_id` (the caller passes `len(real_ocr_words)` so injected ids
never collide with real OCR token ids) and increment sequentially, ONLY for rows
that are actually emitted (skipped points consume no id).

`points_by_class` accepts either shape:
  - dict: {class_label: [(x, y), ...], ...}
  - list: [(x, y, class_label), ...]
Deduping across tiles is assumed already done by the caller (a separate concern,
already solved for the other path in `molmo_candidates.py`) — this function does
NOT re-dedupe points.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

Point = Tuple[float, float]
PointsByClass = Union[Dict[str, Sequence[Point]], Sequence[Tuple[float, float, str]]]
Word = Tuple[str, float, float, float, float]     # text, x0, y0, x1, y1 (image px)
SyntheticRow = Tuple[int, str, float, float]       # id, text, cx_norm, cy_norm

PLACEHOLDER_TEXT = "?"


def _flatten_points(points_by_class: PointsByClass) -> List[Tuple[float, float, str]]:
    """Normalize either accepted shape into a flat [(x, y, class_label), ...] list,
    preserving iteration order (dict insertion order, then list order)."""
    if isinstance(points_by_class, dict):
        flat: List[Tuple[float, float, str]] = []
        for cls, pts in points_by_class.items():
            for (x, y) in pts:
                flat.append((x, y, cls))
        return flat
    return [(x, y, cls) for (x, y, cls) in points_by_class]


def _nearest_word(x: float, y: float, ocr_words: Sequence[Word]):
    """Return (distance, word) for the OCR word whose box CENTER is nearest to
    (x, y), or (None, None) if `ocr_words` is empty."""
    best_word = None
    best_dist = None
    for w in ocr_words:
        text, x0, y0, x1, y1 = w
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if best_dist is None or d < best_dist:
            best_dist = d
            best_word = w
    return best_dist, best_word


def molmo_synthetic_tokens(
    points_by_class: PointsByClass,
    ocr_words: Sequence[Word],
    W: float,
    H: float,
    *,
    near_radius: float = 30.0,
    pair_radius: float = 120.0,
    start_id: int = 0,
) -> List[SyntheticRow]:
    """Build synthetic `idx_words`-shaped rows for Molmo2 points not already
    covered by a real OCR token. `W`, `H` are the rendered page's pixel dimensions
    (needed for the same normalization `ocr_reasoning_extract` uses internally)."""
    rows: List[SyntheticRow] = []
    next_id = start_id
    for (x, y, _cls) in _flatten_points(points_by_class):
        dist, word = _nearest_word(x, y, ocr_words)

        if dist is not None and dist <= near_radius:
            continue  # OCR already has a token here -- skip, don't double-count

        if dist is not None and dist <= pair_radius:
            text = word[0]
        else:
            text = PLACEHOLDER_TEXT

        rows.append((next_id, text, x / W, y / H))
        next_id += 1

    return rows


# --------------------------------------------------------------------------- #
# Unit tests (run directly: `python molmo_synthetic_tokens.py`)
# --------------------------------------------------------------------------- #

def _test_point_near_existing_word_is_skipped():
    ocr_words = [("TAG-1", 100.0, 100.0, 120.0, 112.0)]   # center ~ (110, 106)
    points = {"valve": [(112.0, 108.0)]}                   # ~3.6px from center
    rows = molmo_synthetic_tokens(points, ocr_words, W=1000, H=1000, start_id=5)
    assert rows == [], f"expected point near an OCR word to be skipped, got {rows}"


def _test_point_far_but_within_pair_radius_borrows_text():
    ocr_words = [("TAG-2", 100.0, 100.0, 120.0, 112.0)]   # center ~ (110, 106)
    # 60px away: outside near_radius(30) but inside pair_radius(120)
    points = {"valve": [(170.0, 106.0)]}
    rows = molmo_synthetic_tokens(points, ocr_words, W=1000, H=500, start_id=5)
    assert len(rows) == 1, rows
    rid, text, cx, cy = rows[0]
    assert rid == 5
    assert text == "TAG-2"
    assert abs(cx - 170.0 / 1000) < 1e-9
    assert abs(cy - 106.0 / 500) < 1e-9


def _test_point_with_nothing_nearby_gets_placeholder():
    ocr_words = [("TAG-3", 100.0, 100.0, 120.0, 112.0)]
    points = {"valve": [(900.0, 900.0)]}   # far outside pair_radius too
    rows = molmo_synthetic_tokens(points, ocr_words, W=1000, H=1000, start_id=0)
    assert len(rows) == 1, rows
    rid, text, cx, cy = rows[0]
    assert rid == 0
    assert text == "?"


def _test_ids_start_at_start_id_and_are_unique_no_collision():
    ocr_words = [("TAG-4", 0.0, 0.0, 10.0, 10.0)]
    # 3 points: one skipped (near TAG-4), two far away (placeholder)
    points_by_class = {
        "valve": [(3.0, 3.0), (900.0, 900.0)],
        "instrument_bubble": [(950.0, 10.0)],
    }
    start_id = 133  # e.g. len(real_ocr_words)
    rows = molmo_synthetic_tokens(points_by_class, ocr_words, W=1000, H=1000, start_id=start_id)
    ids = [r[0] for r in rows]
    assert len(rows) == 2, rows          # the (3,3) point was skipped
    assert all(i >= start_id for i in ids), ids
    assert len(set(ids)) == len(ids), "duplicate ids emitted"
    assert ids == sorted(ids), "ids should increment sequentially in emission order"


def _run_tests():
    tests = [
        _test_point_near_existing_word_is_skipped,
        _test_point_far_but_within_pair_radius_borrows_text,
        _test_point_with_nothing_nearby_gets_placeholder,
        _test_ids_start_at_start_id_and_are_unique_no_collision,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_tests()
