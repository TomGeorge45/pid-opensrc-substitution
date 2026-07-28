"""Skeletonize a binary mask via Zhang-Suen thinning.

Reduces every connected ink region to a 1-px-wide centerline while
preserving topology (the number of holes, the connectivity between
endpoints, the count and location of branch points).

Uses ``skimage.morphology.skeletonize`` which implements the Zhang-Suen
algorithm by default — pure pixel arithmetic, deterministic, no parameters.
"""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def skeletonize_lines(binary: np.ndarray) -> np.ndarray:
    """Return a (H, W) uint8 skeleton mask (255 = skeleton pixel, 0 = empty).

    Input
    -----
    binary : (H, W) uint8 — output of ``binarize`` + ``mask_symbols``
             (255 = INK, 0 = background).
    """
    if binary.ndim != 2:
        raise ValueError(f"expected (H, W) binary mask, got shape {binary.shape}")
    if binary.dtype != np.uint8:
        raise ValueError(f"expected uint8 input, got {binary.dtype}")
    # skimage expects bool; returns bool.
    skel = skeletonize(binary > 0)
    return (skel.astype(np.uint8) * 255)
