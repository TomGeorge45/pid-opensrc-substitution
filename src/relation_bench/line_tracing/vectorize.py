"""Vectorize a 1-px skeleton into ordered polyline pixel sequences.

Given a skeleton + a list of junction positions, we:

1. Mask out a small zone around each junction position (so the segments
   between junctions become disjoint connected components).
2. Label connected components of the remaining skeleton (8-conn).
3. For each component, find its two endpoints (pixels with exactly 1
   neighbor in the component) and BFS-walk between them in order.
4. Simplify the ordered polyline via Ramer-Douglas-Peucker (industry-
   standard polyline simplification, tolerance defaults to 2 px).

Returns a list of polylines; each polyline is a list of (x, y) tuples in
drawing coordinates. Loose-end skeleton runs (no junction at one end)
still produce a polyline — the driver classifies the endpoint as
``loose_end`` in the final Segment record.

Closed loops (no endpoint pixel) are skipped in V1 — they're rare in P&IDs
(unique to certain instrument loops drawn as a circle).
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np
from scipy import ndimage
from skimage.measure import approximate_polygon

_8_NEIGHBORHOOD = np.ones((3, 3), dtype=np.uint8)
_8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


def walk_segments(
    skeleton: np.ndarray,
    junction_positions: List[Tuple[int, int]],
    *,
    junction_zone_radius_px: int = 4,
    simplify_tolerance_px: float = 2.0,
) -> List[List[Tuple[int, int]]]:
    """Return a list of polylines, each a list of (x, y) tuples in drawing coords.

    Parameters
    ----------
    skeleton
        (H, W) uint8 — 255 where the skeleton is, 0 elsewhere.
    junction_positions
        List of (x, y) junction centers. The walk excludes a small disk
        around each.
    junction_zone_radius_px
        Pixels to clear around each junction before component labeling.
        Slightly larger than detect's cluster_radius_px is fine.
    simplify_tolerance_px
        Ramer-Douglas-Peucker tolerance. 2 px keeps straight pipes as
        2-vertex polylines and most L/U bends at 3-4 vertices.
    """
    if skeleton.ndim != 2:
        raise ValueError(f"expected (H, W) skeleton, got shape {skeleton.shape}")
    h, w = skeleton.shape

    # Mask out a disk around each junction.
    skel_segments = (skeleton > 0).astype(np.uint8)
    for (jx, jy) in junction_positions:
        x0 = max(0, jx - junction_zone_radius_px)
        y0 = max(0, jy - junction_zone_radius_px)
        x1 = min(w, jx + junction_zone_radius_px + 1)
        y1 = min(h, jy + junction_zone_radius_px + 1)
        skel_segments[y0:y1, x0:x1] = 0

    # Label connected components (8-conn).
    labeled, n_components = ndimage.label(skel_segments, structure=_8_NEIGHBORHOOD)
    polylines: List[List[Tuple[int, int]]] = []
    if n_components == 0:
        return polylines

    # Compute neighbor count for each remaining pixel (for endpoint detection).
    neighbor_count = ndimage.convolve(
        skel_segments,
        _8_NEIGHBORHOOD,
        mode="constant",
        cval=0,
    )

    for comp_label in range(1, n_components + 1):
        ys, xs = np.where(labeled == comp_label)
        if len(ys) < 2:
            # Single-pixel component — too short to form a segment.
            continue

        # Endpoints in this component: pixels with exactly 1 neighbor.
        # convolve count includes self → endpoint has count == 2 (self+1 neighbor).
        endpoint_pixels = [
            (int(y), int(x))
            for y, x in zip(ys, xs)
            if int(neighbor_count[y, x]) == 2
        ]

        if len(endpoint_pixels) >= 2:
            # Pick the two endpoints farthest apart (for branching artifacts
            # we still want the longest walk through the component).
            start, end = _farthest_pair(endpoint_pixels)
            ordered = _bfs_path(labeled, comp_label, start, end)
        elif len(endpoint_pixels) == 1:
            # Loose-end segment that runs into a junction or stops. Walk
            # from the endpoint to the farthest reachable pixel.
            start = endpoint_pixels[0]
            ordered = _bfs_path_to_farthest(labeled, comp_label, start)
        else:
            # No endpoints → closed loop. V1 skips these (rare in P&IDs).
            continue

        if len(ordered) < 2:
            continue

        # Simplify via RDP. approximate_polygon expects an (N, 2) array.
        pts = np.array(ordered)  # (y, x)
        simplified = approximate_polygon(pts, tolerance=simplify_tolerance_px)
        if len(simplified) < 2:
            simplified = pts[[0, -1]]
        polylines.append([(int(x), int(y)) for y, x in simplified])

    return polylines


def _farthest_pair(points: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the two points farthest apart in Euclidean distance."""
    best = (points[0], points[1])
    best_d = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dy = points[i][0] - points[j][0]
            dx = points[i][1] - points[j][1]
            d = dy * dy + dx * dx
            if d > best_d:
                best_d = d
                best = (points[i], points[j])
    return best


def _bfs_path(
    labeled: np.ndarray,
    comp_label: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """BFS from ``start`` to ``end`` within the component, returning
    the ordered pixel path. Each pixel is (y, x).
    """
    h, w = labeled.shape
    parent = {start: None}
    q = deque([start])
    while q:
        cy, cx = q.popleft()
        if (cy, cx) == end:
            break
        for dy, dx in _8_OFFSETS:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if labeled[ny, nx] != comp_label:
                continue
            if (ny, nx) in parent:
                continue
            parent[(ny, nx)] = (cy, cx)
            q.append((ny, nx))
    if end not in parent:
        return [start]
    # Walk back from end to start.
    path: List[Tuple[int, int]] = []
    cur: object = end
    while cur is not None:
        path.append(cur)  # type: ignore[arg-type]
        cur = parent[cur]  # type: ignore[index]
    return list(reversed(path))


def _bfs_path_to_farthest(
    labeled: np.ndarray,
    comp_label: int,
    start: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """BFS from ``start`` to the farthest reachable pixel within the component."""
    h, w = labeled.shape
    parent = {start: None}
    farthest = start
    farthest_dist = 0
    dist = {start: 0}
    q = deque([start])
    while q:
        cy, cx = q.popleft()
        for dy, dx in _8_OFFSETS:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if labeled[ny, nx] != comp_label:
                continue
            if (ny, nx) in parent:
                continue
            parent[(ny, nx)] = (cy, cx)
            dist[(ny, nx)] = dist[(cy, cx)] + 1
            if dist[(ny, nx)] > farthest_dist:
                farthest_dist = dist[(ny, nx)]
                farthest = (ny, nx)
            q.append((ny, nx))
    return _bfs_path(labeled, comp_label, start, farthest)
