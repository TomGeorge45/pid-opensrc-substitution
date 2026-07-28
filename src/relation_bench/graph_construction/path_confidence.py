"""Confidence model for traversed paths.

Multiplicative penalty model — every factor in ``[0.5, 1.0]``. A long
mixed path with ambiguous junctions ends up at ~0.5; a short direct path
stays close to the segment's baseline confidence.

The constants are V1 heuristics — see PRODUCTION_TODO.md "Stage 11 —
Calibrate confidence model from labeled paths" for the V1.5+ corpus-based
tuning entry.
"""
from __future__ import annotations

from collections import Counter

from .path_traversal import TraversalPath


def compute_path_confidence(path: TraversalPath) -> float:
    """Combine baseline segment confidence with three multiplicative penalties.

    Factors (all in ``[0.5, 1.0]``):
      - ``base``                = path.avg_segment_confidence
      - ``length_penalty``      = max(0.5, 1.0 - 0.05 * (n_segments - 1))
      - ``type_consistency``    = (count_of_most_common_line_type / n_segments)
      - ``ambiguous_penalty``   = max(0.5, 1.0 - 0.10 * n_ambiguous_junctions)
    """
    n = path.n_segments
    if n == 0:
        return 0.0

    base = max(0.0, min(1.0, path.avg_segment_confidence))
    length_penalty = max(0.5, 1.0 - 0.05 * (n - 1))

    if path.line_types:
        top_count = Counter(path.line_types).most_common(1)[0][1]
        type_consistency = top_count / n
    else:
        type_consistency = 1.0
    type_consistency = max(0.5, type_consistency)

    ambiguous_penalty = max(0.5, 1.0 - 0.10 * path.n_ambiguous_junctions)

    return round(base * length_penalty * type_consistency * ambiguous_penalty, 4)


def dominant_line_type(path: TraversalPath) -> str:
    """Most-common ``line_type`` along the path. Ties → ``unknown``."""
    if not path.line_types:
        return "unknown"
    counts = Counter(path.line_types)
    top = counts.most_common(1)[0]
    if len(counts) > 1 and counts.most_common(2)[1][1] == top[1]:
        return "unknown"
    return top[0]
