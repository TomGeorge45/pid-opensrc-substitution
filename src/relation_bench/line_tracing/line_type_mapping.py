"""Map stroke style → physical line type.

ISA-5.1 + ISO 14617 industry standard:

  continuous  → process flow
  dashed      → electric signal
  dotted      → pneumatic signal
  double_line → capillary
  unknown     → unknown

This mapping is universal across vendors / tenants — encoded here as
INDUSTRY STANDARD, NOT as a tenant-configurable file. Same "no hardcoded
tenant data" rule that governs the grammars utility. Per-tenant overrides
(some EPCs invert the convention) deferred to V1.5+ via OntologyCache,
NOT via a tenant config file.

PROD: See PRODUCTION_TODO.md → "Stage 6 — Per-tenant line-style overlays".
"""
from __future__ import annotations

from typing import Dict

from .models import LineType, StrokeStyle


_STROKE_STYLE_TO_LINE_TYPE: Dict[StrokeStyle, LineType] = {
    "continuous": "process",
    "dashed": "electric_signal",
    "dotted": "pneumatic",
    "double_line": "capillary",
    "unknown": "unknown",
}


def line_type_for_stroke(style: StrokeStyle) -> LineType:
    """Return the canonical ISA-5.1 line type for the given stroke style."""
    return _STROKE_STYLE_TO_LINE_TYPE.get(style, "unknown")
