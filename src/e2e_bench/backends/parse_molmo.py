"""
Molmo2 <points> parser, ported verbatim (logic unchanged) from
pid-ml/src/stage4_symbol_detection/molmo_candidate.py::parse — the parser behind the
recorded Stage 4 detection runs (F1=0.628 at tile=512/upscale=2/enhance=True). Wrapped to
return ParseOutcome[list[NormalizedDetection]] instead of the original (detections, error)
tuple, and to require entity_type be supplied by the caller since Molmo2's native pointing
format carries no type label (see the original docstring: "No native confidence or per-point
label documented in this format").
"""
import re

from ..types import NormalizedDetection, ParseOutcome

_POINTS_RE = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')


def parse_molmo_points(text: str, tile_w: int, tile_h: int, entity_type: str,
                       confidence: float = 0.5) -> ParseOutcome:
    """tile_w/tile_h: the dimensions of the (possibly upscaled) tile the model actually
    saw. Returned bbox_tile is in THAT same tile-local pixel space — the stage04 converter
    is responsible for dividing by the upscale factor before adding the tile origin
    (Conversion_Layer_Plan.md 5.4 step 1), matching the Stage 4 detection notebook's math.

    confidence: Molmo2's format has no native confidence (D3) - fixed default 0.5, never
    0.0 (assembly drops confidence==0.0 records)."""
    detections = []
    for m in _POINTS_RE.finditer(text):
        nums = [float(v) for v in re.split(r'[\t:;, ]+', m.group(1).strip()) if v]
        if len(nums) % 3 != 0:
            if len(nums) >= 2 and nums[0] == nums[1] and (len(nums) - 1) % 3 == 0:
                nums = nums[1:]
            else:
                return ParseOutcome.failed(text, f"coords not a multiple of 3 (frame,x,y): {nums}")
        for i in range(0, len(nums), 3):
            _frame, x_scaled, y_scaled = nums[i:i + 3]
            x, y = x_scaled / 1000 * tile_w, y_scaled / 1000 * tile_h
            detections.append(NormalizedDetection(
                bbox_tile=[x - 20, y - 20, x + 20, y + 20],
                confidence=confidence,
                entity_type=entity_type,
            ))
    if not detections and ("<point" in text.lower()):
        return ParseOutcome.failed(text, "contains point-like tags but regex found no matches")
    return ParseOutcome.ok(detections, text)
