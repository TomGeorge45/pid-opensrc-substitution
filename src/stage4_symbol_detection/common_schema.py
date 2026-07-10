"""Common scoring schema (checklist 2.4) — the fixed target format every candidate's raw
output gets parsed into, regardless of what format the model natively emits.

Deliberately simpler than the real agent's DetectionRecord (see Agent_Pipeline_Facts.md §2) —
benchmarking only needs enough to score detection + typing, not provenance/tag-parsing.

    {
        "bbox": [x0, y0, x1, y1],   # tile-local pixel coords, xyxy, absolute (not normalized)
        "confidence": float | None,  # 0.0-1.0, or None if the model doesn't natively emit one
        "entity_type": str | None,  # typing eval (Kaggle) only; None/ignored for Gupta detection
    }

Point-based candidates (Molmo2) additionally carry a "point": [x, y] key alongside a small
fixed pseudo-bbox centered on the point — real scoring should use point-in-box matching for
these, not IoU against the pseudo-bbox (see CLAUDE.md's point-in-box rule).
"""
