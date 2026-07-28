"""R1 raster line-tracer — thin standalone port of pnid-intelligence-agent Stage 6's
``_process_page`` (agents/pnid-intelligence-agent/pnid_agent/stages/line_tracing/driver.py).

That function is already artifact-store-free and async-free (see driver.py: the async
``stage_06_run`` orchestrator does page I/O, downscaling, timeouts, and JSON writes, but
delegates the actual 10-step CV algorithm to a plain synchronous ``_process_page`` taking
numpy arrays + bbox lists). This module is that same 10-step body, copied verbatim minus
the orchestration this benchmark doesn't need:
  - no Kafka-safe asyncio.wait_for timeout wrapping (the benchmark runs sheets one at a
    time locally; a hang is a bug to see immediately, not something to silently skip)
  - no downscale-then-rescale of oversized pages (PID2Graph sheets are pre-sized; the
    benchmark can add this back if a sheet ever needs it)
  - no ArtifactStore reads/writes; caller passes a grayscale numpy array + symbol bboxes
    in, gets a PageLineGraph back

Since this benchmark scores relation-stage TOPOLOGY (does an edge exist between symbol A
and symbol B), not detection, ``symbol_bboxes``/``symbol_det_ids`` here are the GT node
boxes/ids from pid2graph_gt.SheetGT — the entity stage is given, not run.

``extra_mask_bboxes`` exists because PID2Graph's node ontology has no text/tag class:
prod's real Stage 4 detections include tag/label text boxes, which Stage 6 masks
alongside symbols so tag pixels don't fragment the skeleton into spurious junctions.
PID2Graph gives us none of that, so callers should pass OCR-derived text boxes here to
approximate it (see Benchmark_Gaps_Register.md gap #9-confirmed). These boxes are
masked-only — never added to symbol_bboxes/symbol_det_ids, so endpoints can never snap
to a text box as if it were a real symbol.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .binarize import binarize
from .bridge import bridge_across_symbols
from .endpoints import build_junction_grid, resolve_endpoint
from .junctions import classify_junction, detect_junction_candidates
from .line_type_mapping import line_type_for_stroke
from .mask_symbols import mask_symbols
from .merge import merge_collinear_fragments
from .models import Junction, PageLineGraph, Segment
from .skeletonize import skeletonize_lines
from .stroke_style import classify_stroke_style
from .vectorize import walk_segments

BBox = Tuple[int, int, int, int]


def process_page(
    *,
    page_index: int,
    page_gray: np.ndarray,
    symbol_bboxes: Sequence[BBox],
    symbol_det_ids: Sequence[str],
    extra_mask_bboxes: Sequence[BBox] = (),
    binarize_block_size: int = 51,
    binarize_c: int = 7,
    symbol_mask_pad_px: int = 12,
    junction_cluster_radius_px: int = 5,
    junction_window_px: int = 30,
    bridge_max_gap_px: int = 30,
    junction_snap_radius_px: int = 12,
    symbol_snap_radius_px: int = 60,
    polyline_simplify_tolerance_px: float = 2.0,
    merge_collinear_enabled: bool = True,
    merge_max_join_gap_px: int = 36,
    merge_max_angle_deg: float = 14.0,
) -> PageLineGraph:
    """Run all 10 algorithmic steps for one page and return its line graph.

    Defaults match Stage 6's own defaults (driver.py's ``stage_06_run`` signature) so a
    zero-config call reproduces prod behavior at full resolution.
    """
    binary = binarize(page_gray, block_size=binarize_block_size, c=binarize_c)
    masked = mask_symbols(binary, symbol_bboxes, pad_px=symbol_mask_pad_px)
    if extra_mask_bboxes:
        # Masked out (so text doesn't fragment the skeleton into spurious junctions)
        # but NOT added to symbol_bboxes/symbol_det_ids below — endpoints must never
        # snap to a text box as if it were a real symbol.
        masked = mask_symbols(masked, extra_mask_bboxes, pad_px=2)
    skeleton = skeletonize_lines(masked)

    candidates = detect_junction_candidates(
        skeleton, cluster_radius_px=junction_cluster_radius_px,
    )
    junctions_kept: List[Tuple[int, int, str, bool]] = []
    for (cx, cy) in candidates:
        result = classify_junction(skeleton, cx, cy, window_px=junction_window_px)
        if result is None:
            continue
        kind, ambiguous = result
        junctions_kept.append((cx, cy, kind, ambiguous))

    junctions_kept.sort(key=lambda jx: (jx[1], jx[0]))
    junction_positions = [(jx[0], jx[1]) for jx in junctions_kept]
    junction_ids = [f"p{page_index}_j{idx:04d}" for idx in range(len(junctions_kept))]

    polylines = walk_segments(
        skeleton,
        junction_positions,
        junction_zone_radius_px=junction_cluster_radius_px,
        simplify_tolerance_px=polyline_simplify_tolerance_px,
    )

    bridged = bridge_across_symbols(
        polylines,
        list(symbol_bboxes),
        list(symbol_det_ids),
        max_gap_px=bridge_max_gap_px,
    )

    if merge_collinear_enabled:
        bridged = merge_collinear_fragments(
            bridged,
            junction_positions,
            max_join_gap_px=merge_max_join_gap_px,
            max_angle_deg=merge_max_angle_deg,
            junction_block_radius_px=junction_snap_radius_px,
        )

    junction_grid = build_junction_grid(
        junction_positions, junction_ids, cell_size_px=junction_snap_radius_px,
    )
    segments_out: List[Segment] = []
    junction_incidence: Dict[str, List[str]] = {jid: [] for jid in junction_ids}

    bridged_sorted = sorted(bridged, key=lambda pp: (pp[0][0][1], pp[0][0][0]))
    for seg_idx, (poly, passed_through) in enumerate(bridged_sorted):
        if len(poly) < 2:
            continue
        seg_id = f"p{page_index}_g{seg_idx:04d}"
        style, conf = classify_stroke_style(poly, page_gray)
        line_type = line_type_for_stroke(style)
        ep_a = resolve_endpoint(
            poly[0], junction_grid, symbol_bboxes, symbol_det_ids,
            junction_snap_radius_px=junction_snap_radius_px,
            symbol_snap_radius_px=symbol_snap_radius_px,
            cell_size_px=junction_snap_radius_px,
        )
        ep_b = resolve_endpoint(
            poly[-1], junction_grid, symbol_bboxes, symbol_det_ids,
            junction_snap_radius_px=junction_snap_radius_px,
            symbol_snap_radius_px=symbol_snap_radius_px,
            cell_size_px=junction_snap_radius_px,
        )
        if ep_a.kind == "junction" and ep_a.ref:
            junction_incidence.setdefault(ep_a.ref, []).append(seg_id)
        if ep_b.kind == "junction" and ep_b.ref:
            junction_incidence.setdefault(ep_b.ref, []).append(seg_id)
        segments_out.append(
            Segment(
                segment_id=seg_id,
                page_index=page_index,
                polyline=[(int(p[0]), int(p[1])) for p in poly],
                stroke_style=style,
                line_type=line_type,
                endpoint_a=ep_a,
                endpoint_b=ep_b,
                passes_through_symbols=passed_through,
                confidence=round(conf, 4),
            )
        )

    junctions_out: List[Junction] = []
    for (jx, jy, kind, ambiguous), jid in zip(junctions_kept, junction_ids):
        junctions_out.append(
            Junction(
                junction_id=jid,
                page_index=page_index,
                position=(jx, jy),
                kind=kind,
                ambiguous=ambiguous,
                incident_segment_ids=sorted(set(junction_incidence.get(jid, []))),
            )
        )

    stats: Dict[str, int] = {
        "segments_total": len(segments_out),
        "junctions_total": len(junctions_out),
        "ambiguous_junctions": sum(1 for j in junctions_out if j.ambiguous),
        "loose_end_segments": sum(
            1
            for s in segments_out
            if s.endpoint_a.kind == "loose_end" or s.endpoint_b.kind == "loose_end"
        ),
    }
    for s in segments_out:
        stats[f"segments_by_line_type_{s.line_type}"] = (
            stats.get(f"segments_by_line_type_{s.line_type}", 0) + 1
        )
    for j in junctions_out:
        stats[f"junctions_by_kind_{j.kind}"] = (
            stats.get(f"junctions_by_kind_{j.kind}", 0) + 1
        )

    return PageLineGraph(
        page_index=page_index,
        segments=segments_out,
        junctions=junctions_out,
        stats=stats,
    )
