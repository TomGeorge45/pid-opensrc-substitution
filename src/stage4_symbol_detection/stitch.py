"""Phase 3.3 — tile -> sheet stitch + dedup.

Input is tiled with overlap (checklist 2.3), so a boundary symbol can appear in 2+ tiles.
The real agent produces sheet-level output after cross-tile NMS (Agent_Pipeline_Facts.md
§1-2). Scoring raw per-tile predictions double-counts boundary symbols and distorts
precision/recall vs. what the agent actually emits — so this must run before any scoring.

Dedup rule: fair to both box and point predictions (same 3.2 decision as scoring.py) —
cluster by proximity of the unified match-point (box center or native point), not IoU,
since IoU isn't defined for a point prediction.
"""

from .scoring import prediction_point

DEDUP_DISTANCE_PX = 30  # two predictions of the same type within this distance are one symbol


def remap_tile_to_sheet(pred, tile_origin):
    """pred: common-schema dict with tile-local bbox (+ optional point). tile_origin: (ox, oy).
    Returns a new dict with sheet-coordinates, matching the real agent's tile_to_drawing
    convention (Agent_Pipeline_Facts.md §1: bbox_drawing = [x0+ox, y0+oy, x1+ox, y1+oy])."""
    ox, oy = tile_origin
    out = dict(pred)
    x0, y0, x1, y1 = pred["bbox"]
    out["bbox"] = [x0 + ox, y0 + oy, x1 + ox, y1 + oy]
    if pred.get("point") is not None:
        px, py = pred["point"]
        out["point"] = [px + ox, py + oy]
    return out


def stitch_and_dedup(tile_predictions, tile_origins, dedup_distance_px=DEDUP_DISTANCE_PX):
    """tile_predictions: list of per-tile prediction lists (one list per tile, tile-local
    coords). tile_origins: parallel list of (ox, oy) per tile. Returns a single
    sheet-level list of predictions, deduplicated.

    Dedup keeps the highest-confidence prediction in each cluster (ties broken by first-seen).
    Predictions with no confidence (Molmo2) are never preferred over ones that have it, but
    still get deduplicated among themselves.
    """
    assert len(tile_predictions) == len(tile_origins)

    sheet_preds = []
    for preds, origin in zip(tile_predictions, tile_origins):
        for p in preds:
            sheet_preds.append(remap_tile_to_sheet(p, origin))

    # cluster by (entity_type, proximity) — mirrors real per-type dedup, since two different
    # symbol types shouldn't merge just because they're spatially close
    clusters = []  # each: {"type": ..., "members": [pred, ...]}
    for pred in sheet_preds:
        pt = prediction_point(pred)
        etype = pred.get("entity_type")
        placed = False
        for cluster in clusters:
            if cluster["type"] != etype:
                continue
            cx, cy = prediction_point(cluster["members"][0])
            if ((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2) ** 0.5 <= dedup_distance_px:
                cluster["members"].append(pred)
                placed = True
                break
        if not placed:
            clusters.append({"type": etype, "members": [pred]})

    deduped = []
    for cluster in clusters:
        members = cluster["members"]
        with_conf = [m for m in members if m.get("confidence") is not None]
        best = max(with_conf, key=lambda m: m["confidence"]) if with_conf else members[0]
        deduped.append(best)

    return deduped, len(sheet_preds)
