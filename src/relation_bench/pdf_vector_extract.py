"""Part B — extract raw line segments from a born-digital PDF page, the input
`line_tracing.build_vector_page_graph` expects (`segments_px`, a flat list of
`(x0, y0, x1, y1)` tuples in raster pixel space).

This is the piece PID2Graph can never exercise (it ships PNGs only, confirmed via direct
dataset-search verification — see Benchmark_Gaps_Register.md Group 3, gap #5). The AG/RIVE
sheets are real, born-digital PDFs (confirmed via direct PyMuPDF inspection: thousands of
real vector drawing paths per page, not a scanned raster wrapped in PDF), so this is the
first time in this project the vector fast-path — PR #711's actual headline finding — can
be exercised at all.

PyMuPDF's `page.get_drawings()` returns a list of paths, each a dict with an `items` list
of draw commands (`("l", p1, p2)` line, `("c", p1, p2, p3, p4)` bezier curve, `("re", rect)`
rectangle, `("qu", quad)` quad). We flatten every command to its endpoint-defining straight
segment(s) — curves are chorded (start->end only; P&ID pipe curves are rare and a chord is a
reasonable topology-preserving approximation, not a shape-fidelity claim), rectangles/quads
become their 4 boundary edges. Degenerate (near-zero-length) segments are dropped, matching
`build_vector_page_graph`'s own `> 1.0` length filter (redundant but cheap).

Rotation matters here and is easy to get silently wrong: these sheets are rotated 270°
(confirmed live on both AG/RIVE test sheets), and `page.get_pixmap` — what prod's own
`rasterize.render_page` calls, and therefore what every real Tag's `bbox_px` is measured
against — auto de-rotates. Every point here goes through `page.rotation_matrix` before
scaling, the same transform prod's own hybrid-token code applies to vector text
(`extract.py`'s `PNID_HYBRID_TOKENS` branch: `fitz.Rect(...) * rmat) * zoom`) — without it,
segment coordinates land in a DIFFERENT space than the real tag bboxes, silently misaligning
every symbol resolution downstream. Page width/height are likewise measured post-rotation
(swapped for a 90°/270° page, not `page.rect.width/height` directly).
"""
from __future__ import annotations

from typing import List, Tuple

import fitz  # PyMuPDF

Seg = Tuple[float, float, float, float]


def _tp(p, rmat, scale: float) -> Tuple[float, float]:
    """Transform a raw PDF point through rotation then scale — same order as prod's own
    vector-text transform (`(rect * rmat) * zoom`)."""
    q = fitz.Point(p.x, p.y) * rmat
    return (q.x * scale, q.y * scale)


def extract_page_vector_segments(pdf_path: str, page_index: int = 0,
                                  *, render_scale: float = 1.0) -> Tuple[List[Seg], Tuple[int, int]]:
    """Returns (segments_px, (page_width_px, page_height_px)) in the SAME rotated,
    scaled pixel space as `rasterize.render_page`'s output and every real Tag's `bbox_px`.

    `render_scale` must match the zoom the caller rendered the page raster at (e.g. prod's
    own `work_zoom`) — PDF native coordinates are in points (1/72in); `render_scale` here is
    applied directly to (already-rotated) points.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    rmat = page.rotation_matrix
    # PAGE SIZE: use page.rect DIRECTLY — it is already the *displayed* (rotation-applied)
    # rect, so multiplying it by rotation_matrix rotates a second time and returns the page
    # TRANSPOSED. Verified on PX-2368-0180004-001 (2026-07-27): mediabox 792x1224 (portrait),
    # page.rect 1224x792 (landscape, already rotated), `page.rect * rmat` 792x1224 (wrong),
    # and `page.get_pixmap(zoom=5)` — what prod's rasterize.render_page calls, and therefore
    # what every real Tag.bbox_px is measured against — is 6120x3960, i.e. page.rect * zoom.
    # Individual POINTS from get_drawings()/get_text() ARE in unrotated space and DO need rmat
    # (confirmed: the embedded word "MBD-0100" transforms to exactly tag t0002's bbox
    # [745,110,914,145]); it is only the rect that must not be re-rotated.
    # This was previously latent rather than harmful — the sole consumer, diagram_area_bbox,
    # only reads `max(page_size[0], page_size[1])`, which is transposition-invariant — but it
    # would silently corrupt anything that treats width and height distinctly.
    w_px = int(round(abs(page.rect.width) * render_scale))
    h_px = int(round(abs(page.rect.height) * render_scale))

    segments: List[Seg] = []
    for path in page.get_drawings():
        for item in path.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = _tp(item[1], rmat, render_scale), _tp(item[2], rmat, render_scale)
                segments.append((p1[0], p1[1], p2[0], p2[1]))
            elif kind == "c":
                # Bezier: chord start->end only (see module docstring).
                p1, p4 = _tp(item[1], rmat, render_scale), _tp(item[4], rmat, render_scale)
                segments.append((p1[0], p1[1], p4[0], p4[1]))
            elif kind in ("re", "qu"):
                rect_or_quad = item[1]
                if kind == "qu":
                    pts = [_tp(getattr(rect_or_quad, attr), rmat, render_scale)
                          for attr in ("ul", "ur", "lr", "ll")]
                else:
                    x0, y0, x1, y1 = rect_or_quad
                    corners = [fitz.Point(x0, y0), fitz.Point(x1, y0),
                               fitz.Point(x1, y1), fitz.Point(x0, y1)]
                    pts = [_tp(c, rmat, render_scale) for c in corners]
                for k in range(4):
                    a, b = pts[k], pts[(k + 1) % 4]
                    segments.append((a[0], a[1], b[0], b[1]))

    doc.close()
    return [
        s for s in segments
        if ((s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2) ** 0.5 > 1.0
    ], (w_px, h_px)
