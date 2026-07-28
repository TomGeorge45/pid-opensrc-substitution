"""Part B — render the agreement-diff's disagreements on the real sheet, for human
spot-check. Same technique that resolved the earlier v1-vs-zero-shot Molmo2 mystery this
session: don't trust the metric alone, look at what's actually drawn.

Draws three categories of pair, color-coded:
  - green  = agree (LLM claimed it, geometry traced it too)
  - red    = llm_only (LLM claimed connectivity, geometry didn't trace it)
  - blue   = geometry_only (geometry traced a connection, LLM didn't claim it)
Each pair is a line between the two tags' bbox centers, with small circle markers and the
tag id labeled at each endpoint, on top of the real rendered PDF page.
"""
from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Set, Tuple

import fitz
from PIL import Image, ImageDraw, ImageFont

_COLORS = {"agree": (30, 180, 60), "llm_only": (220, 40, 40), "geometry_only": (40, 90, 220)}


def render_agreement_overlay(
    pdf_path: str,
    tags_by_id: Dict[str, dict],
    render_dpi: int,
    categories: Dict[str, Set[FrozenSet[str]]],
    out_path: str,
    *,
    max_pairs_per_category: int = 15,
    downsample_max_px: int = 2400,
    seed: int = 0,
) -> None:
    zoom = render_dpi / 72.0
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB" if pix.n < 4 else "RGBA", (pix.width, pix.height), pix.samples)
    if img.mode != "RGB":
        img = img.convert("RGB")
    doc.close()

    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    def center(tid: str):
        bbox = tags_by_id.get(tid, {}).get("bbox_px") or []
        if len(bbox) != 4:
            return None
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    legend_lines = []
    for cat_name, pairs in categories.items():
        color = _COLORS[cat_name]
        sample = list(pairs)
        rng.shuffle(sample)
        sample = sample[:max_pairs_per_category]
        drawn = 0
        for pair in sample:
            a, b = tuple(pair)
            ca, cb = center(a), center(b)
            if ca is None or cb is None:
                continue
            draw.line([ca, cb], fill=color, width=4)
            for pt, tid in ((ca, a), (cb, b)):
                r = 10
                draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], outline=color, width=3)
                draw.text((pt[0] + r + 2, pt[1] - r), tid, fill=color)
            drawn += 1
        legend_lines.append(f"{cat_name}: {drawn}/{len(pairs)} shown (color {color})")

    y = 10
    for line in legend_lines:
        draw.text((10, y), line, fill=(0, 0, 0))
        y += 24

    w, h = img.size
    scale = min(1.0, downsample_max_px / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(out_path)
