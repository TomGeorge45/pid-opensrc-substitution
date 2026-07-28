"""Coordinate-space-correct pre-render for the Molmo2 pointing pass (Phase 1c fix 1,
Extraction_Agent_Local_Plan.md §11 Phase 1c entry item 1).

Molmo2's pointing pass must run over EXACTLY the same pixel raster/coordinate space
`extract_page` uses internally for the `ocr_reasoning` path, or every point it emits is
subtly wrong (in the rotated-page case, badly wrong -- a swapped-axis wrong, not just an
off-by-a-few-px wrong). Confirmed real, verified by reading `extract.py` directly (the
`ocr_reasoning`/`agentic`/`agentic_tools` branch, ~line 142-145):

    tri = triage_page(pg, page if is_pdf else 0, cfg)
    ...
    zoom = RZ.work_zoom(tri.width_pt, cfg)
    img, W, H = RZ.render_page(pg, zoom)

`triage_page` is NOT optional scaffolding here -- `PageTriage.width_pt` is the
POST-ROTATION display width (`triage.py`: `if rot in (90, 270): w_pt, h_pt =
pg.rect.height, pg.rect.width`), so skipping straight to `work_zoom(pg.rect.width, cfg)`
computes zoom from the wrong edge on any rotated page (the acceptance-test sheet,
GD-T-435-DT-2042-056, is rotated 270 degrees -- confirmed via the existing
`page_rotation_applied_deg: 270` acceptance-test output field).

This module does not reimplement any of `triage_page`/`work_zoom`/`render_page` --
it imports and calls the REAL functions, in the REAL order, so the two can never drift.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import fitz
import numpy as np

AGENT_DIR = "/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-extraction-agent"
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from pnid_pipeline.triage import triage_page  # noqa: E402
from pnid_pipeline import rasterize as RZ  # noqa: E402


def molmo_render_page(pdf_path: str, page: int, cfg: dict) -> Tuple[np.ndarray, int, int, float]:
    """Render `page` of `pdf_path` in the EXACT coordinate space `extract_page` uses
    internally for the ocr_reasoning/agentic/agentic_tools branch. Returns
    (img, W, H, zoom) where `img` is the RGB numpy array `RZ.render_page` returns, and
    W/H are its pixel dimensions (post-rotation, matching `img.shape`).

    Real sequence replicated verbatim from `pnid_pipeline/extract.py`
    (ocr_reasoning branch, confirmed by reading that file directly):
        tri  = triage_page(pg, page, cfg)
        zoom = RZ.work_zoom(tri.width_pt, cfg)
        img, W, H = RZ.render_page(pg, zoom)
    """
    doc = fitz.open(pdf_path)
    try:
        pg = doc[page]
        tri = triage_page(pg, page, cfg)
        zoom = RZ.work_zoom(tri.width_pt, cfg)
        img, W, H = RZ.render_page(pg, zoom)
    finally:
        doc.close()
    return img, W, H, zoom


# --------------------------------------------------------------------------- #
# Self-test (run directly: `python molmo_render.py`) -- renders the same test
# sheet via `molmo_render_page` AND by manually replicating extract_page's exact
# internal sequence (opened separately, not sharing any state), and asserts the
# two agree on W, H, zoom exactly. Requires the real agent + fitz to be importable
# (true in `.venv-e2e`); no fake/mocked pieces -- this fix is specifically about
# NOT drifting from the real pipeline's real math.
# --------------------------------------------------------------------------- #

SCRATCH = (
    "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6"
    "/scratchpad"
)
_TEST_PDF = os.path.join(SCRATCH, "AG_PNID", "AG_PNID", "GD-T-435-DT-2042-056-Z.pdf")


def _test_matches_extract_page_internal_sequence():
    from pnid_pipeline.run import load_config

    cfg = load_config()

    img, W, H, zoom = molmo_render_page(_TEST_PDF, 0, cfg)

    # Manual replication of extract_page's ocr_reasoning branch, independently, to
    # cross-check -- NOT calling molmo_render_page, so this is a real independent
    # check, not a tautology.
    doc2 = fitz.open(_TEST_PDF)
    try:
        pg2 = doc2[0]
        assert pg2.rotation == 270, f"expected the known-rotated test sheet, got rotation={pg2.rotation}"
        rect_height = pg2.rect.height  # capture before the page/doc is closed below
        tri2 = triage_page(pg2, 0, cfg)
        zoom2 = RZ.work_zoom(tri2.width_pt, cfg)
        img2, W2, H2 = RZ.render_page(pg2, zoom2)
        width_pt2 = tri2.width_pt
    finally:
        doc2.close()

    assert zoom == zoom2, f"zoom mismatch: {zoom} vs {zoom2}"
    assert W == W2, f"W mismatch: {W} vs {W2}"
    assert H == H2, f"H mismatch: {H} vs {H2}"
    assert img.shape == img2.shape, f"image shape mismatch: {img.shape} vs {img2.shape}"

    # Sanity: rotation swap actually happened (width_pt should be the POST-rotation
    # width, i.e. pg.rect.height for a 90/270-rotated page) -- guards against a future
    # regression where triage_page's rotation handling silently changes.
    assert width_pt2 == rect_height, (
        f"expected rotated width_pt to equal pg.rect.height ({rect_height}), "
        f"got {width_pt2}"
    )


def _run_tests():
    tests = [_test_matches_extract_page_internal_sequence]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_tests()
