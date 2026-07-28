"""R0 — resolve a real symbol EXTENT from a seed point (Pipeline 3 v2's new first stage).

This is the stage that unblocks Tier-1 #3, and it's worth being precise about *why* it was
blocked, because the fix is not a better algorithm — it's a better input.

Tier-1 #3 asked: given the PDF, work out which drawn shape is each entity's symbol. That was
abandoned as genuinely blocked, and the reasons were real:
  - `closePath` is False on all 5,608 tested vector paths, even for genuinely closed shapes,
    so "is this a closed outline?" is not answerable from the flag.
  - Path bbox sizes span sub-pixel text glyphs to full-page border rectangles, so no global
    size threshold separates equipment outline / pipe segment / text stroke.
Both statements are still true. What changed is that we no longer have to search globally:

  Probe 1 — symbol-extent reconstruction from vector geometry — PASSED, including the harder
  branching case. Tier-1 #3 — the same task — was BLOCKED. The single difference is that
  Probe 1 knew roughly where to look. Molmo2's point is that seed.

Seeded, the question becomes "which of the handful of paths enclosing THIS point is the
symbol", and the size band that was useless globally becomes decisive locally.

Molmo2 is the right seed source, and this is measured rather than assumed — frozen 20-sheet
Gupta test set, class-agnostic detection: Molmo2-points F1 **0.6276** (P 0.6309 / R 0.6244)
vs GPT-5.5-low F1 **0.5125** (P 0.5335 / R 0.4932). The frontier model is *worse* here, because
Molmo2 has a native pointing head while the others emit coordinates as text tokens. A frontier
model therefore cannot substitute as a reference entity source either — and worse, its errors
are systematically the tag-text-vs-symbol confusion, i.e. the exact variable under test, so
using it as a reference would contaminate the control group.

Every resolved extent records HOW it was obtained (`extent_source`) and a confidence, so
score-time stratification can answer "are our numbers limited by extent quality?" instead of
leaving R0 as an unmeasured black box in the middle of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from entities import BBox, EntitySet, Point, SymbolNode

# A candidate path must enclose the seed and sit inside this size band to be believable as a
# symbol outline. Expressed in inches of real page space so it travels across sheets rendered
# at different zooms (same discipline as the old EQUIPMENT_BBOX_PAD_INCHES, which this
# replaces). Deliberately wide: a small ISA bubble is ~0.25in, a large vessel outline can be
# several inches. The band's job is only to reject the two pathological ends — sub-glyph
# strokes and the page border rect — which is exactly what an unseeded search could not do.
MIN_SYMBOL_INCHES = 0.10
MAX_SYMBOL_INCHES = 4.0

# A path whose bbox covers more than this fraction of the page is structural (border, title
# block frame, drawing-area rect), never a symbol.
MAX_PAGE_AREA_FRAC = 0.06

# Fallback box half-size when no plausible enclosing path is found, in inches. Crude by
# design and flagged as `radius_fallback` so it can be excluded from any headline number.
FALLBACK_HALF_INCHES = 0.22


@dataclass(frozen=True)
class VectorPath:
    """One PDF drawing path, kept WHOLE (unlike pdf_vector_extract, which flattens every path
    to loose segments for the tracer). R0 needs path identity and path bbox — that's the
    grouping the tracer deliberately throws away."""
    bbox: BBox
    n_items: int
    closed_hint: bool          # `closePath` as reported; unreliable, kept for diagnostics only

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def contains(self, p: Point, *, pad: float = 0.0) -> bool:
        x, y = p
        x0, y0, x1, y1 = self.bbox
        return (x0 - pad) <= x <= (x1 + pad) and (y0 - pad) <= y <= (y1 + pad)


def extract_page_vector_paths(
    pdf_path: str, page_index: int = 0, *, render_scale: float = 1.0,
) -> Tuple[List[VectorPath], Tuple[int, int]]:
    """Every drawing path on the page WITH its bbox, in the same rotated+scaled pixel space as
    `pdf_vector_extract.extract_page_vector_segments` and every real Tag's `bbox_px`.

    The rotation handling here mirrors `pdf_vector_extract` deliberately — these sheets are
    rotated 270 degrees, `get_pixmap` auto-de-rotates, and skipping `rotation_matrix` silently
    lands coordinates in a different space than the tag bboxes, misaligning everything
    downstream. Same transform, same order: rotate, then scale.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        rmat = page.rotation_matrix
        # page.rect is ALREADY rotation-applied — do not re-rotate it (that transposes the
        # page). Points from get_drawings() are in unrotated space and DO need rmat. See the
        # long note in pdf_vector_extract.extract_page_vector_segments for the verification.
        w_px = int(round(abs(page.rect.width) * render_scale))
        h_px = int(round(abs(page.rect.height) * render_scale))

        paths: List[VectorPath] = []
        for path in page.get_drawings():
            rect = path.get("rect")
            if rect is None:
                continue
            r = (fitz.Rect(rect) * rmat)
            x0, y0 = r.x0 * render_scale, r.y0 * render_scale
            x1, y1 = r.x1 * render_scale, r.y1 * render_scale
            bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            paths.append(VectorPath(
                bbox=bbox,
                n_items=len(path.get("items", [])),
                closed_hint=bool(path.get("closePath", False)),
            ))
        return paths, (w_px, h_px)
    finally:
        doc.close()


def resolve_extent_from_seed(
    seed: Point,
    paths: Sequence[VectorPath],
    *,
    render_dpi: int,
    page_size: Tuple[int, int],
    snap_pad_px: float = 3.0,
    fallback_bbox: Optional[BBox] = None,
) -> Tuple[Optional[BBox], str, float]:
    """Pick the drawn shape that owns `seed`. Returns (extent, extent_source, confidence).

    The rule, in order:
      1. Keep only paths whose bbox encloses the seed (this is the step the unseeded version
         of this problem could not perform, and it typically cuts thousands of candidates to
         single digits).
      2. Reject anything outside the plausible symbol size band, and anything covering more
         than MAX_PAGE_AREA_FRAC of the page — that's the border/title-block frame.
      3. Among survivors take the SMALLEST by area: the tightest shape enclosing the point is
         the most specific one, the same tie-break `hierarchy/priors.py` already uses for
         choosing a containing parent.

    Confidence is deliberately simple and interpretable rather than tuned: how decisively the
    winner beat the runner-up on area (a clear winner is a confident resolution; two similar
    enclosing shapes means we may have picked the wrong one). Nothing downstream gates on it
    yet — it exists so extent quality is measurable, per this module's docstring.
    """
    min_side = MIN_SYMBOL_INCHES * render_dpi
    max_side = MAX_SYMBOL_INCHES * render_dpi
    page_area = max(1.0, float(page_size[0]) * float(page_size[1]))

    candidates = [
        p for p in paths
        if p.contains(seed, pad=snap_pad_px)
        and min_side <= max(p.width, p.height) <= max_side
        and p.width > 0 and p.height > 0
        and (p.area / page_area) <= MAX_PAGE_AREA_FRAC
    ]

    if not candidates:
        # Prefer the entity's OWN supplied bbox over an invented circle. Measured basis
        # (2026-07-27, all 3 AG/RIVE sheets): instruments and valves arrive from extraction at
        # ~58x42, which is roughly their real drawn symbol, whereas the radius fallback is an
        # arbitrary 0.44in square. So when no plausible enclosing path exists, the supplied box
        # is strictly better information than a guess.
        #
        # This matters far more than it looks. Across the 3 sheets the dominant R0 failure is
        # "no path within the size band" — 64/68 on GD-B-540, 163/163 on PX-2365 — because those
        # larger sheets' CAD exporters group many strokes into region-sized path objects, so the
        # smallest path enclosing a seed is far bigger than any symbol. A better SEED cannot fix
        # that; there is no tight path to find. Falling back to the supplied box keeps those
        # symbols usable instead of replacing real measurements with a circle.
        if fallback_bbox is not None and len(tuple(fallback_bbox)) == 4:
            return tuple(fallback_bbox), "given_bbox", 0.0
        half = FALLBACK_HALF_INCHES * render_dpi
        x, y = seed
        return (x - half, y - half, x + half, y + half), "radius_fallback", 0.0

    candidates.sort(key=lambda p: p.area)
    winner = candidates[0]
    # Containment is tested with `snap_pad_px` tolerance, so a winning path's bbox can sit up to
    # that far from the seed and fail `EntitySet.validate()`'s strict "seed inside extent" check
    # (observed as 2-3 violations on GD-B-540 and PX-2368). Expand minimally to include the seed
    # rather than loosening the invariant — an extent that does not contain its own seed is a real
    # inconsistency, not a cosmetic one.
    wx0, wy0, wx1, wy1 = winner.bbox
    sx, sy = seed
    if not (wx0 <= sx <= wx1 and wy0 <= sy <= wy1):
        winner = VectorPath(
            bbox=(min(wx0, sx), min(wy0, sy), max(wx1, sx), max(wy1, sy)),
            n_items=winner.n_items, closed_hint=winner.closed_hint)
    if len(candidates) == 1:
        conf = 1.0
    else:
        runner = candidates[1]
        # ratio in (0,1]; small winner vs much larger runner-up => decisive => high confidence
        conf = 1.0 - (winner.area / runner.area) if runner.area > 0 else 0.5
        conf = max(0.0, min(1.0, conf))
    return winner.bbox, "vector_seeded", round(float(conf), 4)


def resolve_extents(
    entity_set: EntitySet,
    pdf_path: str,
    *,
    render_dpi: int,
    page_index: int = 0,
    only_types: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Resolve extents in place for every symbol carrying a seed point. Returns a summary
    {extent_source: count} for logging/stratification.

    `only_types` scopes the work — pass `{"equipment"}` to run the equipment-only variant the
    diagnostic recommends. That recommendation is not arbitrary: all 5 measured wrong-endpoint
    failures on PX-2368 have a wrong EQUIPMENT end and a correct instrument end, because
    instrument bubbles arrive at ~58x42 (about right) while equipment arrives at ~120x20 (a
    name plate). So equipment is where nearly all of the available win is.

    `overwrite=False` leaves an already-resolved extent alone — so a hand-corrected extent
    (`extent_source="hand"`) is never silently replaced by an automatic guess.
    """
    zoom = render_dpi / 72.0
    paths, page_size = extract_page_vector_paths(pdf_path, page_index, render_scale=zoom)

    wanted = set(only_types) if only_types is not None else None
    summary: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    for s in entity_set.symbols:
        if s.point is None:
            continue
        if wanted is not None and (s.type not in wanted):
            continue
        if s.extent is not None and not overwrite and s.extent_source == "hand":
            counts["hand"] = counts.get("hand", 0) + 1
            continue
        extent, source, conf = resolve_extent_from_seed(
            s.point, paths, render_dpi=render_dpi, page_size=page_size,
            fallback_bbox=s.extent)
        s.extent = extent
        s.extent_source = source
        s.extent_conf = conf
        counts[source] = counts.get(source, 0) + 1

    summary = {k: str(v) for k, v in sorted(counts.items())}
    return summary


def seed_from_bbox_center(entity_set: EntitySet, *, only_types: Optional[Iterable[str]] = None) -> int:
    """Bootstrap seeds from the CENTRE of whatever bbox an entity already has.

    Honest about what this is: for instruments and valves, whose supplied bbox is roughly the
    real symbol, the centre lands inside the shape and the seed is legitimate. For EQUIPMENT it
    lands inside the *name plate*, which may sit well away from the drawn shape — MBD-0100's
    plate is 131x25 px at (2929,1141) while its ellipse is elsewhere entirely. So a
    bbox-centre seed on equipment will often resolve to the text's own path, not the vessel.

    Provided only so the pipeline can run end-to-end before Molmo2 retraining lands, and so
    the A/B has a mechanical baseline to beat. It is NOT a substitute for a real pointing
    model or a hand-placed seed on equipment. Returns the number of seeds set.
    """
    wanted = set(only_types) if only_types is not None else None
    n = 0
    for s in entity_set.symbols:
        if s.point is not None or s.extent is None:
            continue
        if wanted is not None and (s.type not in wanted):
            continue
        x0, y0, x1, y1 = s.extent
        s.point = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        n += 1
    return n
