"""Render annotated crops (red box=A, blue box=B, texts labeled) for the DELTA pairs the
Tier-1 upgrades made vs original, so Opus can adjudicate whether the upgrades helped.

Two delta categories per sheet:
  - backbone_added = upgraded - original  (edges the backbone pass found by walking fittings)
  - line_removed   = original - upgraded  (edges dropped because they touched a line-label tag)
"""
import json
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

SCRATCH = "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/10ffbddb-0f7d-4f19-a544-f1152513500c/scratchpad"
manifest = json.load(open(f"{SCRATCH}/r4_bundle/manifest.json"))

SHEET_PDF = {
    "PX-2368-0180004-001": f"{SCRATCH}/sheets/RIVE/PX-2368-0180004-001.pdf",
    "GD-B-540-DP-2920-005-Z": f"{SCRATCH}/sheets/AG_PNID/GD-B-540-DP-2920-005-Z.pdf",
    "PX-2365-0140006-001": f"{SCRATCH}/sheets/RIVE/PX-2365-0140006-001.PDF",
}

sheet_id = sys.argv[1] if len(sys.argv) > 1 else "PX-2368-0180004-001"
n_per_cat = int(sys.argv[2]) if len(sys.argv) > 2 else 6

OUT = Path(f"{SCRATCH}/adjudicate/{sheet_id}")
OUT.mkdir(parents=True, exist_ok=True)

s = manifest["sheets"][sheet_id]
render_dpi = s["render_dpi"]
ents = s["entities"]
orig = {tuple(p) for p in s["original_pairs"]}
upg = {tuple(p) for p in s["upgraded_pairs"]}
backbone_added = sorted(upg - orig)
line_removed = sorted(orig - upg)

# need entity bboxes -> reload from the extraction json
pred = json.load(open(f"{SCRATCH}/preds_gpt55_partB_2026-07-24/{sheet_id}_gpt55low.json"))
bbox_by_id = {t["id"]: t["bbox_px"] for t in pred["tags"]}

zoom = render_dpi / 72.0
doc = fitz.open(SHEET_PDF[sheet_id])
pg = doc[0]
pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
page_img = Image.frombytes("RGB" if pix.n < 4 else "RGBA",
                           (pix.width, pix.height), pix.samples).convert("RGB")
doc.close()

MARGIN = 300
MAXPX = 1000

def render(pair, tag):
    a, b = pair
    ba, bb = bbox_by_id.get(a), bbox_by_id.get(b)
    if not ba or not bb:
        return None
    ux0 = min(ba[0], bb[0]); uy0 = min(ba[1], bb[1])
    ux1 = max(ba[2], bb[2]); uy1 = max(ba[3], bb[3])
    cx0 = max(0, int(ux0 - MARGIN)); cy0 = max(0, int(uy0 - MARGIN))
    cx1 = min(page_img.width, int(ux1 + MARGIN)); cy1 = min(page_img.height, int(uy1 + MARGIN))
    crop = page_img.crop((cx0, cy0, cx1, cy1)).copy()
    draw = ImageDraw.Draw(crop)
    for bx, color in ((ba, (220, 20, 20)), (bb, (20, 60, 220))):
        draw.rectangle([bx[0]-cx0, bx[1]-cy0, bx[2]-cx0, bx[3]-cy0], outline=color, width=6)
    w, h = crop.size
    scale = min(1.0, MAXPX / max(w, h))
    if scale < 1.0:
        crop = crop.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    name = f"{tag}__{a}_{b}.png"
    crop.save(OUT / name)
    return name, ents.get(a, {}).get("text"), ents.get(b, {}).get("text"), \
        ents.get(a, {}).get("type"), ents.get(b, {}).get("type")

print(f"=== {sheet_id} ===")
print(f"backbone_added: {len(backbone_added)}, line_removed: {len(line_removed)}\n")

print("BACKBONE-ADDED (red=A blue=B) — are these real drawn connections?")
for pair in backbone_added[:n_per_cat]:
    r = render(pair, "added")
    if r:
        print(f"  {r[0]}  {r[1]}({r[3]}) <-> {r[2]}({r[4]})")

print("\nLINE-REMOVED — were these spurious connections to a pipe-label tag?")
for pair in line_removed[:n_per_cat]:
    r = render(pair, "removed")
    if r:
        print(f"  {r[0]}  {r[1]}({r[3]}) <-> {r[2]}({r[4]})")

print(f"\ncrops -> {OUT}")
