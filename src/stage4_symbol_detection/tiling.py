"""Checklist 2.3 tiling scheme, extracted from Stage4_Phase2_Data_Preparation.ipynb into a
reusable module (originally only lived as a notebook cell — duplicated here so 5.2/5.3
training can tile Gupta sheets the same way eval/inference does, rather than training on raw
full sheets that don't match real usage).

Exact params from Agent_Pipeline_Facts.md §1: 1024x1024 tiles, 205px overlap, 819px stride,
uniform grid, edge tiles clamped smaller (not repositioned). No title-block carve-out (see
Stage4_Checklist_Status.md item 2.3 for why — this repo has no title-block detector).
"""

TILE_SIZE = 1024
OVERLAP = 205
STRIDE = TILE_SIZE - OVERLAP  # 819


def compute_tile_grid(img_w, img_h, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Uniform grid, edge tiles clamped smaller (not repositioned) — matches grid.py:41-94."""
    stride = tile_size - overlap
    tiles = []
    y0 = 0
    while y0 < img_h:
        y1 = min(y0 + tile_size, img_h)
        x0 = 0
        while x0 < img_w:
            x1 = min(x0 + tile_size, img_w)
            is_edge = (x1 - x0 < tile_size) or (y1 - y0 < tile_size)
            tiles.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "is_edge": is_edge})
            x0 += stride
        y0 += stride
    return tiles


def yolo_line_to_xyxy(line, img_w, img_h):
    parts = line.split()
    cls = parts[0]
    cx, cy, w, h = (float(v) for v in parts[1:5])
    cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    return cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def boxes_intersect(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def tile_and_remap(img_path, label_path):
    """Returns (img, tiles, orig_boxes). Each tile dict gains "boxes_tile": list of
    (cls, x0, y0, x1, y1) in tile-local coords — a box is assigned to EVERY tile it
    intersects (boundary symbols correctly appear in overlapping tiles), not clipped to
    tile bounds."""
    from PIL import Image

    img = Image.open(img_path)
    W, H = img.size
    tiles = compute_tile_grid(W, H)

    orig_boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if line:
                orig_boxes.append(yolo_line_to_xyxy(line, W, H))

    for t in tiles:
        tbox = (t["x0"], t["y0"], t["x1"], t["y1"])
        t["boxes_tile"] = []
        for cls, x0, y0, x1, y1 in orig_boxes:
            if boxes_intersect((x0, y0, x1, y1), tbox):
                t["boxes_tile"].append((cls, x0 - t["x0"], y0 - t["y0"], x1 - t["x0"], y1 - t["y0"]))

    return img, tiles, orig_boxes
