"""Detect and classify junctions in a 1-px skeleton.

A *junction* is a skeleton pixel where >2 8-connected neighbors are also
skeleton pixels — that's where lines branch or cross. We:

1. Find all such candidate pixels.
2. Cluster nearby candidates (within ``cluster_radius_px``) into one
   junction position — a fat junction can produce several adjacent
   candidates.
3. Classify each junction by looking at a ``window_px`` × ``window_px``
   window around it and counting "stubs" (connected components of
   skeleton pixels at the window perimeter):

   - 2 stubs → not a real junction, drop it (artifact of cluster merging)
   - 3 stubs → ``tee``
   - 4 stubs → ``cross`` (V1; ``ambiguous=True`` so V1.5+ Sonnet tiebreaker
     can flip it to ``jumper`` later)
   - ≥5 stubs → ``joint`` (multi-way confluence)

PROD:
  - V1 cannot reliably distinguish ``cross`` from ``jumper`` purely from
    pixels — a hand-drawn jumper might or might not leave a detectable
    gap after binarization + skeletonization. We default the 4-stub case
    to ``cross`` (conservative — avoids inventing connections that aren't
    there) and mark it ``ambiguous=True``. V1.5+ adds a Sonnet tiebreaker
    on a 60×60 crop with a single Yes/No question. See PRODUCTION_TODO.md
    → "Stage 6 — VLM tiebreaker on ambiguous cross-vs-jumper".
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .models import JunctionKind


_8_NEIGHBORHOOD = np.ones((3, 3), dtype=np.uint8)


def detect_junction_candidates(
    skeleton: np.ndarray,
    *,
    cluster_radius_px: int = 5,
) -> List[Tuple[int, int]]:
    """Return deduplicated ``(x, y)`` junction-candidate positions.

    Two stages:
      1. Find every skeleton pixel whose 8-connected neighbor count >2.
      2. Cluster candidates within ``cluster_radius_px`` (so a fat
         junction merges to a single representative position — the
         component centroid).
    """
    if skeleton.ndim != 2:
        raise ValueError(f"expected (H, W) skeleton, got shape {skeleton.shape}")
    skel_bool = skeleton > 0

    # Count 8-connected skeleton neighbors per pixel. The convolution sums the
    # surrounding 3x3 region (incl. self). Subtract self to get neighbor count.
    neighbor_count = ndimage.convolve(
        skel_bool.astype(np.uint8),
        _8_NEIGHBORHOOD,
        mode="constant",
        cval=0,
    )
    # candidate iff skeleton pixel AND has >2 skeleton neighbors (excluding self).
    candidates = skel_bool & (neighbor_count >= 4)  # 3 neighbors + self in conv sum

    if not candidates.any():
        return []

    # Dilate candidates with a disk of `cluster_radius_px`, then label connected
    # components — every cluster becomes one component, centroid = junction position.
    structure = np.ones(
        (2 * cluster_radius_px + 1, 2 * cluster_radius_px + 1),
        dtype=np.uint8,
    )
    dilated = ndimage.binary_dilation(candidates, structure=structure)
    labeled, n = ndimage.label(dilated, structure=_8_NEIGHBORHOOD)
    if n == 0:
        return []
    centroids = ndimage.center_of_mass(dilated, labeled, range(1, n + 1))
    # center_of_mass returns (y, x); we want (x, y) ints.
    return sorted(
        ((int(round(cx)), int(round(cy))) for cy, cx in centroids),
        key=lambda p: (p[1], p[0]),
    )


def classify_junction(
    skeleton: np.ndarray,
    x: int,
    y: int,
    *,
    window_px: int = 30,
) -> Optional[Tuple[JunctionKind, bool]]:
    """Return ``(kind, ambiguous)`` for a junction centered on ``(x, y)``.

    Returns ``None`` when the candidate isn't a real junction — fewer than
    3 stubs cross the window perimeter (e.g. an L-corner, which the
    candidate detector happens to flag due to 8-connectivity diagonal
    neighborhood, but which carries no topological meaning).

    See module docstring for the heuristic.
    """
    h, w = skeleton.shape
    half = window_px // 2
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(w, x + half + 1)
    y1 = min(h, y + half + 1)
    window = skeleton[y0:y1, x0:x1] > 0
    if window.size == 0:
        return ("joint", False)

    # Perimeter ring: the 1-px-thick border of the window.
    perim_mask = np.zeros_like(window, dtype=bool)
    perim_mask[0, :] = True
    perim_mask[-1, :] = True
    perim_mask[:, 0] = True
    perim_mask[:, -1] = True
    perim_skeleton = window & perim_mask

    # A "stub" is a connected component of perimeter skeleton pixels — that's
    # one line leaving the junction.
    _, n_stubs = ndimage.label(perim_skeleton, structure=_8_NEIGHBORHOOD)

    if n_stubs <= 2:
        # Not a real junction (corner, dead-end, or cluster artifact).
        # The driver drops these.
        return None
    if n_stubs == 3:
        return ("tee", False)
    if n_stubs == 4:
        # PROD: V1 cannot reliably tell cross vs jumper from pixels alone.
        # Default to cross (conservative — avoids inventing connections).
        # V1.5+ Sonnet tiebreaker flips ambiguous cases.
        return ("cross", True)
    # ≥5 stubs → multi-way joint.
    return ("joint", False)
