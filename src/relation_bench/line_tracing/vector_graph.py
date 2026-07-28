"""Vector fast-path: build a Stage-6 PageLineGraph from PDF vector segments.

For born-digital P&IDs, Stage 0 harvests exact pipe geometry into
``page.vector_geometry`` (projected to raster px). This module turns those raw
segments into the SAME ``PageLineGraph`` the classical-CV pipeline produces, so
Stage 11 / the asset-flow backbone consume it unchanged.

Why each step exists (the raster pipeline gets these for free; vector must do
them explicitly):

  1. **bridge** — reconnect collinear pipe ends separated by an inline symbol
     (valve/instrument drawn over the line), mirroring ``bridge_across_symbols``.
  2. **planar noding** — CAD often draws a branch tee into the *interior* of a
     through-pipe (no shared endpoint). Split segments at interior touches so
     the tee becomes a real graph node. Without this the graph is false-sparse.
  3. **endpoint merge (union-find)** — coincident endpoints become one node.
  4. **symbol resolution** — a node inside a Stage-4 detection bbox is that
     symbol's connection point (resolved to its ``detection_id``).
  5. **degree-2 contraction** — collapse polyline-vertex chains (degree-2
     junctions) so two assets aren't separated by 20+ hops (mirrors RDP
     simplification on the raster side).

Resolved against **Stage-4 detection ids/bboxes** (Stage 6 runs before Stage 10),
NOT entity temp_ids.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Endpoint, Junction, PageLineGraph, Segment
from .geometry_helpers import bbox_center_outside, diagram_area_bbox

Seg = Tuple[float, float, float, float]


# ── geometry helpers ─────────────────────────────────────────────────────────
def _pt_seg(p, a, b) -> Tuple[float, float]:
    """Distance from point p to segment ab + the foot parameter t (0=a,1=b)."""
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    tc = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + tc * dx), py - (ay + tc * dy)), t


def _grid(bboxes, cell: float) -> Dict[Tuple[int, int], List[int]]:
    g: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        for cx in range(int(x0 // cell), int(x1 // cell) + 1):
            for cy in range(int(y0 // cell), int(y1 // cell) + 1):
                g[(cx, cy)].append(i)
    return g


def _seg_intersects_rect(p1: Tuple[float, float], p2: Tuple[float, float],
                          rect: Tuple[float, float, float, float]) -> bool:
    """Liang-Barsky segment/axis-aligned-rect clip test — True if any part of segment
    p1->p2 lies within rect (touching counts)."""
    x0, y0, x1, y1 = rect
    ax, ay = p1
    bx, by = p2
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return t0 <= t1


def _resolve_passthrough_symbols(
    poly: List[Tuple[float, float]],
    dets: List[Tuple[str, Tuple[float, float, float, float]]],
    exclude_ids: set,
    *,
    pad: float = 12.0,
) -> List[str]:
    """detection_ids whose padded bbox this segment's polyline genuinely crosses through,
    excluding ids already resolved as this segment's own endpoints (those are handled by
    normal symbol-node resolution, not inline pass-through)."""
    if len(poly) < 2:
        return []
    pxs = [p[0] for p in poly]
    pys = [p[1] for p in poly]
    poly_bb = (min(pxs), min(pys), max(pxs), max(pys))
    hits: List[str] = []
    for det_id, bb in dets:
        if det_id in exclude_ids:
            continue
        rect = (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)
        if poly_bb[2] < rect[0] or poly_bb[0] > rect[2] or poly_bb[3] < rect[1] or poly_bb[1] > rect[3]:
            continue  # cheap reject before the precise per-edge test
        for i in range(len(poly) - 1):
            if _seg_intersects_rect(poly[i], poly[i + 1], rect):
                hits.append(det_id)
                break
    return hits


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _bridge_collinear(seg: List[Seg], *, min_gap: float, max_gap: float, min_cos: float) -> List[Seg]:
    """Add bridge segments between open endpoints whose outward directions point
    at each other and are roughly collinear within a gap — reconnects pipes
    interrupted by an inline symbol drawn over the line."""
    eps = []  # (x, y, outx, outy, seg_idx)
    for si, (ax, ay, bx, by) in enumerate(seg):
        L = math.hypot(bx - ax, by - ay) or 1.0
        ux, uy = (bx - ax) / L, (by - ay) / L
        eps.append((ax, ay, -ux, -uy, si))
        eps.append((bx, by, ux, uy, si))
    g = defaultdict(list)
    for i, e in enumerate(eps):
        g[(int(e[0] // 50), int(e[1] // 50))].append(i)

    # Only LOOSE ends bridge — an endpoint shared with another segment is a
    # continuation/junction, not a gap. Bridging those duplicates intervening
    # segments (raises junction degree, blocks contraction). An endpoint is open
    # iff no other endpoint sits within `min_gap` of it.
    def _is_open(i: int) -> bool:
        xi, yi = eps[i][0], eps[i][1]
        cx, cy = int(xi // 50), int(yi // 50)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for j in g.get((cx + ox, cy + oy), []):
                    if j != i and math.hypot(xi - eps[j][0], yi - eps[j][1]) <= min_gap:
                        return False
        return True

    open_flags = [_is_open(i) for i in range(len(eps))]
    bridges: List[Seg] = []
    used = set()
    for i, (xi, yi, dxi, dyi, si) in enumerate(eps):
        if i in used or not open_flags[i]:
            continue
        cx, cy = int(xi // 50), int(yi // 50)
        best, bestgap = None, 1e18
        for ox in range(-3, 4):
            for oy in range(-3, 4):
                for j in g.get((cx + ox, cy + oy), []):
                    if j == i or j in used or not open_flags[j]:
                        continue
                    xj, yj, dxj, dyj, sj = eps[j]
                    if sj == si:
                        continue
                    gx, gy = xj - xi, yj - yi
                    gap = math.hypot(gx, gy)
                    if gap <= min_gap or gap > max_gap:
                        continue
                    gux, guy = gx / gap, gy / gap
                    if (dxi * gux + dyi * guy) > min_cos and (dxj * -gux + dyj * -guy) > min_cos:
                        if gap < bestgap:
                            best, bestgap = j, gap
        if best is not None:
            bridges.append((xi, yi, eps[best][0], eps[best][1]))
            used.add(i); used.add(best)
    return bridges


# ── main builder ─────────────────────────────────────────────────────────────
def build_vector_page_graph(
    page_index: int,
    segments_px: List[Seg],
    symbol_bboxes: Sequence[Tuple[int, int, int, int]],
    symbol_det_ids: Sequence[str],
    *,
    page_size: Optional[Tuple[int, int]] = None,
    node_merge_tol_px: float = 4.0,
    split_tol_px: float = 2.0,
    interior_lo: float = 0.06,
    interior_hi: float = 0.94,
    bridge_enabled: bool = True,
    bridge_max_gap_px: float = 130.0,
    bridge_min_cos: float = 0.96,
    symbol_pad_px: float = 12.0,
    diagram_clip_enabled: bool = True,
) -> PageLineGraph:
    """Build a PageLineGraph from raw vector segments (raster px)."""
    seg: List[Seg] = [
        (ax, ay, bx, by)
        for (ax, ay, bx, by) in segments_px
        if math.hypot(bx - ax, by - ay) > 1.0
    ]

    # 0. diagram-area clip — drop title-block / spec-table grid lines.
    if diagram_clip_enabled and len(symbol_bboxes) >= 8:
        area = diagram_area_bbox(list(symbol_bboxes), page_size=page_size)
        seg = [
            (ax, ay, bx, by) for (ax, ay, bx, by) in seg
            if not bbox_center_outside(
                ((ax + bx) / 2, (ay + by) / 2, (ax + bx) / 2, (ay + by) / 2), area
            )
        ]

    # 1. bridge collinear gaps across inline symbols.
    if bridge_enabled:
        seg = seg + _bridge_collinear(
            seg, min_gap=node_merge_tol_px, max_gap=bridge_max_gap_px, min_cos=bridge_min_cos
        )

    # 2. planar noding: split segments where another endpoint touches the interior.
    endpoints = [(s[0], s[1]) for s in seg] + [(s[2], s[3]) for s in seg]
    # endpoints indexed: a-of-seg-i at i, b-of-seg-i at len(seg)+i
    n = len(seg)
    ep_bb = [(x - split_tol_px, y - split_tol_px, x + split_tol_px, y + split_tol_px) for x, y in endpoints]
    epg = _grid(ep_bb, 50.0)
    noded: List[Seg] = []
    for si, (ax, ay, bx, by) in enumerate(seg):
        x0 = min(ax, bx) - split_tol_px; x1 = max(ax, bx) + split_tol_px
        y0 = min(ay, by) - split_tol_px; y1 = max(ay, by) + split_tol_px
        cand = set()
        for cx in range(int(x0 // 50), int(x1 // 50) + 1):
            for cy in range(int(y0 // 50), int(y1 // 50) + 1):
                cand.update(epg.get((cx, cy), []))
        ts = []
        for ei in cand:
            if ei == si or ei == si + n:  # this segment's own endpoints
                continue
            d, t = _pt_seg(endpoints[ei], (ax, ay), (bx, by))
            if d <= split_tol_px and interior_lo < t < interior_hi:
                ts.append(round(t, 4))
        if not ts:
            noded.append((ax, ay, bx, by)); continue
        ts = sorted(set(ts))
        pts = [(ax, ay)] + [(ax + t * (bx - ax), ay + t * (by - ay)) for t in ts] + [(bx, by)]
        for k in range(len(pts) - 1):
            noded.append((pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]))

    # 3. union-find endpoint merge into nodes.
    nep = [(s[0], s[1]) for s in noded] + [(s[2], s[3]) for s in noded]
    m = len(noded)
    nb = [(x - node_merge_tol_px, y - node_merge_tol_px, x + node_merge_tol_px, y + node_merge_tol_px)
          for x, y in nep]
    ng = _grid(nb, node_merge_tol_px * 2)
    uf = _UF(len(nep))
    for cell, idxs in ng.items():
        cx, cy = cell
        neigh: List[int] = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                neigh.extend(ng.get((cx + ddx, cy + ddy), []))
        for i in idxs:
            xi, yi = nep[i]
            for j in neigh:
                if j <= i:
                    continue
                if math.hypot(xi - nep[j][0], yi - nep[j][1]) <= node_merge_tol_px:
                    uf.union(i, j)
    root_pts: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for i, p in enumerate(nep):
        root_pts[uf.find(i)].append(p)
    root_pos = {r: (sum(x for x, _ in ps) / len(ps), sum(y for _, y in ps) / len(ps))
                for r, ps in root_pts.items()}

    # 4. resolve nodes to symbols (smallest containing Stage-4 detection bbox).
    dets = [
        (det_id, bb, (bb[2] - bb[0]) * (bb[3] - bb[1]))
        for det_id, bb in zip(symbol_det_ids, symbol_bboxes)
    ]
    dg = _grid([(b[0] - symbol_pad_px, b[1] - symbol_pad_px, b[2] + symbol_pad_px, b[3] + symbol_pad_px)
                for _, b, _ in dets], 100.0)
    node_symbol: Dict[int, str] = {}
    for r, (nx, ny) in root_pos.items():
        cx, cy = int(nx // 100), int(ny // 100)
        cand = set()
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                cand.update(dg.get((cx + ddx, cy + ddy), []))
        best, best_area = None, 1e18
        for di in cand:
            det_id, bb, ar = dets[di]
            if bb[0] - symbol_pad_px <= nx <= bb[2] + symbol_pad_px and \
               bb[1] - symbol_pad_px <= ny <= bb[3] + symbol_pad_px and ar < best_area:
                best, best_area = det_id, ar
        if best is not None:
            node_symbol[r] = best

    # 5. multigraph + degree-2 contraction.
    edges: Dict[int, dict] = {}
    inc: Dict[int, set] = defaultdict(set)
    eid = 0
    for k in range(m):
        ra, rb = uf.find(k), uf.find(m + k)
        if ra == rb:
            continue
        edges[eid] = {"u": ra, "v": rb, "poly": [nep[k], nep[m + k]]}
        inc[ra].add(eid); inc[rb].add(eid); eid += 1

    def _other(e, r):
        return e["v"] if e["u"] == r else e["u"]

    work = deque(r for r in inc if r not in node_symbol and len(inc[r]) == 2)
    while work:
        x = work.popleft()
        if x in node_symbol or len(inc[x]) != 2:
            continue
        e1id, e2id = tuple(inc[x])
        e1, e2 = edges[e1id], edges[e2id]
        u, w = _other(e1, x), _other(e2, x)
        if u == w:
            continue
        p1 = e1["poly"] if e1["v"] == x else list(reversed(e1["poly"]))
        p2 = e2["poly"] if e2["u"] == x else list(reversed(e2["poly"]))
        inc[u].discard(e1id); inc[w].discard(e2id)
        inc.pop(x, None); edges.pop(e1id, None); edges.pop(e2id, None)
        edges[eid] = {"u": u, "v": w, "poly": p1[:-1] + p2}
        inc[u].add(eid); inc[w].add(eid); eid += 1
        for nn in (u, w):
            if nn not in node_symbol and len(inc[nn]) == 2:
                work.append(nn)

    # 6. build PageLineGraph.
    jid_of: Dict[int, str] = {}
    jc = 0
    for r in inc:
        if r in node_symbol or len(inc[r]) < 2:
            continue
        jid_of[r] = f"p{page_index}_j{jc:04d}"; jc += 1

    def _endpoint(r) -> Endpoint:
        pos = (int(root_pos[r][0]), int(root_pos[r][1]))
        if r in node_symbol:
            return Endpoint(kind="symbol", ref=node_symbol[r], position=pos)
        if r in jid_of:
            return Endpoint(kind="junction", ref=jid_of[r], position=pos)
        return Endpoint(kind="loose_end", ref=None, position=pos)

    # detection list for inline pass-through resolution below — built once, reused per segment.
    det_list = list(zip(symbol_det_ids, symbol_bboxes))

    segs_model: List[Segment] = []
    ji_inc: Dict[str, List[str]] = defaultdict(list)
    gi = 0
    for e in edges.values():
        poly = [(int(x), int(y)) for x, y in e["poly"]]
        if len(poly) < 2:
            continue
        sid = f"p{page_index}_g{gi:04d}"; gi += 1
        ep_a, ep_b = _endpoint(e["u"]), _endpoint(e["v"])
        # Populate passes_through_symbols (2026-07-28 fix): a straight vector line that crosses
        # a symbol's bbox without ever splitting a node there (CAD draws one continuous line
        # through an inline valve/instrument, unlike the raster path where masking chops the
        # pipe and bridge_across_symbols reconnects it) previously left this ALWAYS empty here —
        # confirmed 0 populated of 8,105 segments across the corpus check, making build_relations
        # pass 0's inline-chain entirely dead code on the vector path. Root cause of symbols with
        # zero relations despite traced pipe running straight through them (measured 12 of 12
        # checked). Excludes this segment's own resolved endpoints (already handled as ordinary
        # symbol-node connections, not inline pass-through).
        exclude = {r for r in (ep_a.ref, ep_b.ref) if r}
        passed = _resolve_passthrough_symbols(poly, det_list, exclude, pad=symbol_pad_px)
        segs_model.append(Segment(
            segment_id=sid, page_index=page_index, polyline=poly,
            stroke_style="continuous", line_type="process",
            endpoint_a=ep_a, endpoint_b=ep_b,
            passes_through_symbols=passed,
            confidence=0.9,
        ))
        for r in (e["u"], e["v"]):
            if r in jid_of:
                ji_inc[jid_of[r]].append(sid)
    junctions = [
        Junction(junction_id=jid, page_index=page_index,
                 position=(int(root_pos[r][0]), int(root_pos[r][1])),
                 kind="joint", ambiguous=False, incident_segment_ids=ji_inc.get(jid, []))
        for r, jid in jid_of.items()
    ]
    stats = {
        "segments_total": len(segs_model),
        "junctions_total": len(junctions),
        "symbol_nodes": len(node_symbol),
        "source": 1,  # 1 = vector fast-path (vs 0 raster); int-only per model
    }
    return PageLineGraph(page_index=page_index, segments=segs_model,
                         junctions=junctions, stats=stats)
