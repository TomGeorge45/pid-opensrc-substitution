"""PaddleOCR adapter matching `pnid_pipeline/vision.py`'s real `tiled_ocr_words`
tiling loop EXACTLY (same tile/overlap math, same `Word` contract), but calling
PaddleOCR 3.7.0 instead of Google Cloud Vision per tile.

`Word = Tuple[str, float, float, float, float]` — (text, x0, y0, x1, y1), image
pixels, mapped back from tile-local coords exactly like `vision.py`'s `gv_words`
callback does for Google Vision.

PaddleOCR 3.7.0's real result shape (verified 2026-07-16 in this project, see
`e2e_bench/backends/parse_paddle.py` and `e2e_harness/poc_run_arm_p_v2.py`'s
`run_paddle_ocr`): `.predict(...)` is the modern unified-pipeline call. It returns
a list of dict-like `OCRResult` objects; ONE per input page/image. Confirmed keys:
`rec_texts` (List[str]), `rec_scores` (List[float]), `rec_boxes`
(List[[x0,y0,x1,y1]], axis-aligned ints, already tile-local pixel coords) — no
quad-polygon parsing needed, simpler than the classic PP-OCRv3/v4 `.ocr()` shape.

Confirmed here (2026-07-17): `.predict()` also accepts an in-memory BGR numpy
array directly (not just a file path), which is what this adapter needs for
per-tile crops — no need to round-trip through disk per tile.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

Word = Tuple[str, float, float, float, float]   # text, x0, y0, x1, y1 (image px)

_OCR_SINGLETON = None


def _get_ocr():
    """Lazy singleton — PaddleOCR construction loads several models; build once,
    reuse across tiles/pages (mirrors the pattern in poc_run_arm_p_v2.run_paddle_ocr,
    just hoisted out of the per-call path since here we call it per TILE, not per page)."""
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        from paddleocr import PaddleOCR
        _OCR_SINGLETON = PaddleOCR(lang="en")
    return _OCR_SINGLETON


def _paddle_words_for_tile(sub: np.ndarray) -> List[Word]:
    """Run PaddleOCR on one BGR tile crop, return TILE-LOCAL word boxes."""
    ocr = _get_ocr()
    result = ocr.predict(sub)
    if not result:
        return []
    page = result[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    boxes = page.get("rec_boxes", [])
    out: List[Word] = []
    for text, _score, box in zip(texts, scores, boxes):
        x0, y0, x1, y1 = (float(v) for v in box)
        out.append((text, x0, y0, x1, y1))
    return out


def paddle_ocr_words(
    img: np.ndarray, *, tile: int = 1400, overlap: int = 220,
) -> List[Word]:
    """Same tiling loop as `pnid_pipeline.vision.tiled_ocr_words` (verbatim math:
    stride = tile - overlap, tiles walk left-to-right/top-to-bottom, last tile in
    each row/col clipped to image bounds), swapping the per-tile OCR engine for
    PaddleOCR. `img` is expected in the SAME layout `tiled_ocr_words` receives
    (RGB-ish array as rendered by `rasterize.render_page`; vision.py converts to
    BGR internally before encoding for Google Vision — PaddleOCR expects BGR too
    so the same cv2.cvtColor conversion is applied here before inference, matching
    vision.py's own per-tile conversion exactly).

    Synchronous (no network round-trip needed, unlike Google Vision) — the real
    `tiled_ocr_words` is async only because it's an HTTP call; this local
    substitute has no I/O to await, so `run_extraction_local.py` wraps it in an
    `async def` shim when monkeypatching `pnid_pipeline.vision.tiled_ocr_words`.
    """
    import cv2

    H, W = img.shape[:2]
    tiles: List[Tuple[int, int, int, int]] = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            tiles.append((x, y, min(x + tile, W), min(y + tile, H)))
            x += max(1, tile - overlap)
        y += max(1, tile - overlap)

    out: List[Word] = []
    for (x0, y0, x1, y1) in tiles:
        sub = img[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        if sub.ndim == 3 and sub.shape[2] >= 3:
            sub = cv2.cvtColor(sub, cv2.COLOR_RGB2BGR)
        words = _paddle_words_for_tile(sub)
        out.extend((t_, x0 + a, y0 + b, x0 + c, y0 + d) for (t_, a, b, c, d) in words)
    return out
