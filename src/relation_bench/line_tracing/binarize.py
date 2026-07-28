"""Binarize a grayscale page raster into a uint8 ink/background mask.

Strategy: adaptive Gaussian threshold (works well across drawings with
uneven brightness or scan artifacts). If the result is degenerate
(<1% or >99% ink), fall back to Otsu's global threshold.

The output uses 255 = INK, 0 = background. That matches what
``skimage.morphology.skeletonize`` expects (truthy = foreground).

PROD: Very-degraded scans (faxes, sub-300-DPI archived blueprints) lose
thin lines through this pipeline. V1.5+ swaps in a learned UNet line
segmentation gated by a per-page heuristic. See PRODUCTION_TODO.md →
"Stage 6 — Learned UNet binarization".
"""
from __future__ import annotations

import cv2
import numpy as np


def binarize(
    page_img_gray: np.ndarray,
    *,
    block_size: int = 51,
    c: int = 7,
) -> np.ndarray:
    """Return a (H, W) uint8 binary mask where 255 = INK, 0 = background.

    Parameters
    ----------
    page_img_gray
        Grayscale page raster (H, W) or (H, W, 1) uint8.
    block_size
        Adaptive-threshold neighborhood size. MUST be odd. Default 51 px
        (~1/8 inch at 400 DPI) is a good fit for engineering drawings.
    c
        Constant subtracted from the neighborhood mean. Higher = more
        conservative (less ink).
    """
    if page_img_gray.ndim == 3:
        page_img_gray = page_img_gray.squeeze(axis=2)
    if page_img_gray.dtype != np.uint8:
        raise ValueError(f"expected uint8 input, got {page_img_gray.dtype}")
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError(f"block_size must be odd and >=3, got {block_size}")

    # Adaptive Gaussian threshold. THRESH_BINARY_INV → ink (dark on page)
    # becomes 255 (foreground in the mask).
    binary = cv2.adaptiveThreshold(
        page_img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )
    if _is_degenerate(binary):
        # Fall back to Otsu. Different lighting / contrast profile but
        # produces a sensible split when adaptive can't.
        _, binary = cv2.threshold(
            page_img_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
    return binary


def _is_degenerate(binary: np.ndarray) -> bool:
    """Return True when the binary mask is mostly all-ink or all-background.

    A real P&ID has between 1% and 30% ink on the page. Outside that range
    the adaptive threshold has likely picked up scan noise or a near-blank
    region; Otsu is a safer fallback.
    """
    ink_fraction = float(np.count_nonzero(binary)) / binary.size
    return ink_fraction < 0.01 or ink_fraction > 0.99
