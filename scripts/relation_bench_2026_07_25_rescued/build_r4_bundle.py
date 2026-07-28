"""LOCAL PREP (CPU, no GPU) for the upgraded-Pipeline-3 benchmark.

Feeds the 3 AG/RIVE sheets' GIVEN entities (GPT-5.5 extraction) through the deterministic
relationship pipeline in BOTH configs (original / upgraded), renders a crop per unique
candidate edge in the v3-relation adapter's trained format (union bbox + bracket coords),
and writes a bundle the GPU notebook consumes for R4 validation.

Produces the PRE-LLM result immediately: original vs upgraded candidate-relation counts on
the real 3 sheets. Post-LLM comes from the notebook.
"""
import json
import sys
import zipfile
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, "/Users/tomgeorge/pid-ml/src/relation_bench")
from relationship_pipeline import run_relationship_pipeline

SCRATCH = "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/10ffbddb-0f7d-4f19-a544-f1152513500c/scratchpad"
PRED_DIR = f"{SCRATCH}/preds_gpt55_partB_2026-07-24"
OUT_DIR = Path(f"{SCRATCH}/r4_bundle")
OUT_DIR.mkdir(exist_ok=True)
CROP_DIR = OUT_DIR / "crops"
CROP_DIR.mkdir(exist_ok=True)

SHEETS = [
    ("PX-2368-0180004-001", f"{PRED_DIR}/PX-2368-0180004-001_gpt55low.json",
     f"{SCRATCH}/sheets/RIVE/PX-2368-0180004-001.pdf"),
    ("GD-B-540-DP-2920-005-Z", f"{PRED_DIR}/GD-B-540-DP-2920-005-Z_gpt55low.json",
     f"{SCRATCH}/sheets/AG_PNID/GD-B-540-DP-2920-005-Z.pdf"),
    ("PX-2365-0140006-001", f"{PRED_DIR}/PX-2365-0140006-001_gpt55low.json",
     f"{SCRATCH}/sheets/RIVE/PX-2365-0140006-001.PDF"),
]

CROP_MARGIN = 250
CROP_MAX_PX = 900

manifest = {"sheets": {}}
summary_rows = []

for sheet_id, json_path, pdf_path in SHEETS:
    d = json.load(open(json_path))
    render_dpi = d["drawing"]["render_dpi"]
    entities = [{"id": t["id"], "text": t.get("text"), "type": t.get("type"),
                 "bbox_px": t.get("bbox_px")} for t in d["tags"]]
    ent_by_id = {e["id"]: e for e in entities}

    orig = run_relationship_pipeline(entities, pdf_path, render_dpi, upgraded=False)
    upg = run_relationship_pipeline(entities, pdf_path, render_dpi, upgraded=True)

    orig_pairs = {(r.a, r.b) for r in orig}
    upg_pairs = {(r.a, r.b) for r in upg}
    all_pairs = sorted(orig_pairs | upg_pairs)

    print(f"{sheet_id}: original={len(orig_pairs)} candidates, upgraded={len(upg_pairs)} "
          f"(+{len(upg_pairs - orig_pairs)} new, -{len(orig_pairs - upg_pairs)} dropped)")
    summary_rows.append((sheet_id, len(orig_pairs), len(upg_pairs),
                         len(upg_pairs - orig_pairs), len(orig_pairs - upg_pairs)))

    # render one crop per unique candidate pair, adapter-format (union bbox + bracket coords)
    zoom = render_dpi / 72.0
    doc = fitz.open(pdf_path)
    pg = doc[0]
    pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    page_img = Image.frombytes("RGB" if pix.n < 4 else "RGBA",
                               (pix.width, pix.height), pix.samples).convert("RGB")
    doc.close()

    candidates = []
    for idx, (a, b) in enumerate(all_pairs):
        ea, eb = ent_by_id.get(a), ent_by_id.get(b)
        if not ea or not eb or not ea.get("bbox_px") or not eb.get("bbox_px"):
            continue
        ba, bb = ea["bbox_px"], eb["bbox_px"]
        ux0 = min(ba[0], bb[0]); uy0 = min(ba[1], bb[1])
        ux1 = max(ba[2], bb[2]); uy1 = max(ba[3], bb[3])
        cx0 = max(0, int(ux0 - CROP_MARGIN)); cy0 = max(0, int(uy0 - CROP_MARGIN))
        cx1 = min(page_img.width, int(ux1 + CROP_MARGIN))
        cy1 = min(page_img.height, int(uy1 + CROP_MARGIN))
        crop = page_img.crop((cx0, cy0, cx1, cy1))
        a_local = [int(ba[0] - cx0), int(ba[1] - cy0), int(ba[2] - cx0), int(ba[3] - cy0)]
        b_local = [int(bb[0] - cx0), int(bb[1] - cy0), int(bb[2] - cx0), int(bb[3] - cy0)]
        # downscale to CROP_MAX_PX, scaling local coords with it
        w, h = crop.size
        scale = min(1.0, CROP_MAX_PX / max(w, h))
        if scale < 1.0:
            crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            a_local = [int(v * scale) for v in a_local]
            b_local = [int(v * scale) for v in b_local]
        crop_name = f"{sheet_id}__{idx:04d}.png"
        crop.save(CROP_DIR / crop_name)
        candidates.append({
            "pair": [a, b], "crop_file": f"crops/{crop_name}",
            "a_local": a_local, "b_local": b_local,
            "a_text": ea.get("text"), "b_text": eb.get("text"),
            "a_type": ea.get("type"), "b_type": eb.get("type"),
            "in_original": (a, b) in orig_pairs, "in_upgraded": (a, b) in upg_pairs,
        })

    manifest["sheets"][sheet_id] = {
        "render_dpi": render_dpi,
        "original_pairs": [list(p) for p in sorted(orig_pairs)],
        "upgraded_pairs": [list(p) for p in sorted(upg_pairs)],
        "candidates": candidates,
        "entities": {e["id"]: {"text": e.get("text"), "type": e.get("type")}
                     for e in entities},
    }

with open(OUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)

# zip the whole bundle for HF push
bundle_zip = Path(f"{SCRATCH}/r4_bundle_2026-07-25.zip")
with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(OUT_DIR / "manifest.json", "manifest.json")
    for crop in sorted(CROP_DIR.glob("*.png")):
        zf.write(crop, f"crops/{crop.name}")

n_crops = len(list(CROP_DIR.glob("*.png")))
print(f"\n=== PRE-LLM candidate counts (deterministic only) ===")
print(f"{'sheet':28} {'orig':>5} {'upg':>5} {'+new':>5} {'-drop':>6}")
for row in summary_rows:
    print(f"{row[0]:28} {row[1]:>5} {row[2]:>5} {row[3]:>5} {row[4]:>6}")
print(f"\nbundle: {n_crops} crops -> {bundle_zip} ({bundle_zip.stat().st_size/1e6:.1f} MB)")
print("manifest written. Ready to push to HF + run the GPU notebook for R4 validation.")
