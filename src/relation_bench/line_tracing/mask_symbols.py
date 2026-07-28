"""Mask out Stage 4 symbol bboxes from the binarized page.

Symbols (pumps, vessels, valves, instruments) are dense ink regions that
break skeletonization — a 50x50 valve symbol would produce dozens of
spurious branches in the skeleton. We paint each symbol's bbox to 0
(background) BEFORE skeletonizing, which leaves the surrounding pipes
intact but discards the symbol's interior pixels.

Step 7 of the algorithm (bridge_across_symbols) reconnects pipes whose
endpoints landed on opposite sides of a masked symbol.

Stage 4's bboxes are sometimes tight against the symbol outline; pad by
``pad_px`` to ensure the full symbol footprint is masked.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1) in drawing coords


def mask_symbols(
    binary: np.ndarray,
    symbol_bboxes: Iterable[Sequence[int]],
    *,
    pad_px: int = 2,
) -> np.ndarray:
    """Return a NEW binary mask with each symbol bbox painted to 0.

    Parameters
    ----------
    binary
        (H, W) uint8 mask from ``binarize`` (255 = INK, 0 = background).
    symbol_bboxes
        Iterable of (x0, y0, x1, y1) bboxes in drawing coords.
    pad_px
        Pixels to expand each bbox in all directions before masking.
        Default 2 px catches outlines Stage 4 may have clipped.
    """
    if binary.ndim != 2:
        raise ValueError(f"expected (H, W) binary mask, got shape {binary.shape}")
    out = binary.copy()
    h, w = out.shape
    for bbox in symbol_bboxes:
        x0, y0, x1, y1 = (int(v) for v in bbox[:4])
        # Pad + clamp to image bounds.
        x0p = max(0, x0 - pad_px)
        y0p = max(0, y0 - pad_px)
        x1p = min(w, x1 + pad_px)
        y1p = min(h, y1 + pad_px)
        if x1p <= x0p or y1p <= y0p:
            continue  # degenerate / fully out-of-bounds
        out[y0p:y1p, x0p:x1p] = 0
    return out
