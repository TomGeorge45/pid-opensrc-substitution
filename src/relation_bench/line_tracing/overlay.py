"""Render a debug overlay PNG: line graph (segments + junctions) over page raster.

Segments are color-coded by ``line_type``; junctions are marker dots
colored by ``kind``. Loose-end segments get a small "X" at the loose end.

Used solely for visual inspection of Stage 6 output — never read back by
downstream stages.
"""
from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw

from .models import Junction, LineType, Segment


_LINE_TYPE_COLORS = {
    "process": (0, 120, 255),         # blue
    "electric_signal": (255, 140, 0),  # orange
    "pneumatic": (220, 60, 220),       # magenta
    "capillary": (100, 200, 100),      # green
    "hydraulic": (200, 50, 50),        # red
    "unknown": (160, 160, 160),        # gray
}

_JUNCTION_KIND_COLORS = {
    "cross": (255, 80, 80),     # red
    "tee": (80, 200, 80),       # green
    "jumper": (255, 200, 0),    # yellow
    "joint": (60, 60, 220),     # navy
}


def draw_line_graph_overlay(
    page_img: Image.Image,
    segments: Sequence[Segment],
    junctions: Sequence[Junction],
) -> Image.Image:
    """Return a NEW RGBA image with the line graph drawn over the page raster."""
    base = page_img.convert("RGB").copy()
    draw = ImageDraw.Draw(base)

    # Segments first so junction markers sit on top.
    for seg in segments:
        color = _LINE_TYPE_COLORS.get(seg.line_type, _LINE_TYPE_COLORS["unknown"])
        # Pillow's `line` takes a flat list of x,y; we pass our polyline as is.
        draw.line(seg.polyline, fill=color, width=2)

    for j in junctions:
        x, y = j.position
        color = _JUNCTION_KIND_COLORS.get(j.kind, (0, 0, 0))
        radius = 6
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=2,
        )
        if j.ambiguous:
            # Inner dot to flag ambiguous.
            draw.ellipse(
                (x - 2, y - 2, x + 2, y + 2),
                fill=color,
            )

    return base
