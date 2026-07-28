"""Classify a polyline's stroke style by sampling intensity along its path.

We sample the ORIGINAL (pre-binarized) grayscale image at evenly-spaced
points along the polyline. The intensity profile tells us:

- **Continuous** — one long dark run, no gaps.
- **Dashed** — alternating dark/light runs of comparable length.
- **Dotted** — many short dark runs separated by gaps (dots have shorter
  run-length than dashes).
- **Double_line** — sampling perpendicular to the polyline reveals two
  parallel dark runs (not yet implemented in V1 — sampling along the
  polyline alone cannot distinguish this; we return ``continuous`` here
  and let Stage 11 / HITL catch it).

V1 keeps this simple: run-length analysis on the binarized profile.
Returns ``(style, confidence)`` where confidence < 0.7 marks the
classification as borderline.

PROD: V1.5+ moves to autocorrelation / FFT for fine-grained dashed vs
dotted disambiguation. See PRODUCTION_TODO.md → "Stage 6 — More
sophisticated stroke-style classification".
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .models import StrokeStyle


def classify_stroke_style(
    polyline: List[Tuple[int, int]],
    page_img_gray: np.ndarray,
    *,
    sample_every_px: int = 3,
    ink_threshold: int = 128,
) -> Tuple[StrokeStyle, float]:
    """Return ``(style, confidence)`` for the polyline.

    ``page_img_gray`` is the ORIGINAL grayscale image (not the skeleton).
    """
    if page_img_gray.ndim == 3:
        page_img_gray = page_img_gray.squeeze(axis=2)
    if len(polyline) < 2:
        return ("unknown", 0.0)

    profile = _sample_along_polyline(polyline, page_img_gray, sample_every_px)
    if profile.size == 0:
        return ("unknown", 0.0)

    # Convert to ink (1) / background (0).
    binary = (profile < ink_threshold).astype(np.uint8)
    runs = _run_lengths(binary)
    if not runs:
        return ("unknown", 0.0)

    # Total samples and ink fraction.
    n_samples = binary.size
    ink_samples = int(binary.sum())
    ink_fraction = ink_samples / n_samples

    # Continuous: one long dark run dominating the profile.
    dark_runs = [length for value, length in runs if value == 1]
    light_runs = [length for value, length in runs if value == 0]

    if not dark_runs:
        return ("unknown", 0.0)

    longest_dark = max(dark_runs)
    n_dark_runs = len(dark_runs)
    avg_dark = sum(dark_runs) / n_dark_runs

    # Heuristic decisions. Order matters.
    if ink_fraction > 0.85 and n_dark_runs <= 2:
        return ("continuous", min(0.95, ink_fraction))

    if n_dark_runs >= 3:
        # Multiple ink runs interleaved with background. Distinguish dotted
        # (many tiny ink spots) from dashed (longer ink runs) by the ratio
        # of average dark-run length to average light-run length:
        #   dotted ≈ avg_dark < 2 px AND avg_light >> avg_dark
        #   dashed ≈ avg_dark ≈ avg_light, both several px
        avg_light = (sum(light_runs) / len(light_runs)) if light_runs else 0
        if avg_dark <= 2 and avg_light >= avg_dark * 2:
            return ("dotted", 0.7)
        if ink_fraction > 0.25:
            return ("dashed", 0.75)
        # Few short dark runs, otherwise mostly white → likely dotted with
        # under-sampling, but mark low confidence.
        return ("dotted", 0.55)

    if ink_fraction < 0.2:
        return ("unknown", 0.3)

    # Default: mostly continuous but uncertain.
    return ("continuous", 0.5)


def _sample_along_polyline(
    polyline: List[Tuple[int, int]],
    image: np.ndarray,
    sample_every_px: int,
) -> np.ndarray:
    """Walk each polyline segment in ``sample_every_px`` increments,
    sampling the image at each step. Returns a 1-D intensity profile.
    """
    h, w = image.shape
    samples: List[int] = []
    for (x0, y0), (x1, y1) in zip(polyline[:-1], polyline[1:]):
        seg_len = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5))
        n_steps = max(1, seg_len // sample_every_px)
        for i in range(n_steps + 1):
            t = i / n_steps if n_steps > 0 else 0.0
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            if 0 <= x < w and 0 <= y < h:
                samples.append(int(image[y, x]))
    return np.array(samples, dtype=np.int16)


def _run_lengths(binary: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (value, length) pairs from a 1-D 0/1 array."""
    if binary.size == 0:
        return []
    out: List[Tuple[int, int]] = []
    cur_val = int(binary[0])
    cur_len = 1
    for v in binary[1:]:
        v = int(v)
        if v == cur_val:
            cur_len += 1
        else:
            out.append((cur_val, cur_len))
            cur_val = v
            cur_len = 1
    out.append((cur_val, cur_len))
    return out
