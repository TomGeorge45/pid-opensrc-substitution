"""Build the Molmo2 pointing LoRA training dataset (V1: Gupta-only, class-agnostic).

Runs LOCALLY on the Mac (CPU prep local + HF, GPU only trains — per project convention).

What it does:
  1. Downloads Gupta (gupta_pid/PID_Dataset.zip) from the private HF datasets repo.
  2. HARD-ASSERTS the 20 frozen test stems (notebooks/stage4/gupta_test_sheets) are
     excluded from training — CLAUDE.md rule 7, checked before every training run.
  3. Tiles every train sheet with the EXACT inference preprocessing from
     src/extraction_local/molmo_points.py: 512px tiles, 102px overlap grid, 2x LANCZOS
     upscale, autocontrast-grayscale enhance. Training distribution == inference
     distribution, or the adapter tunes for images it will never see.
  4. Target per tile: the GT symbol centers (YOLO boxes whose center falls in the tile),
     stored STRUCTURED (list of (x,y) in 0-1000 coords of the upscaled tile) — the
     training script renders them into Molmo's native `<points coords="..."/>` text at
     train time, AFTER probing the base model's real output shape (no raw sample of
     Molmo2's native emission survives locally, so the exact template is verified on GPU
     against a live generation, not guessed here).
  5. Keeps every symbol-bearing tile; samples empty tiles to ~25% of the mix (teaches
     "nothing here" without drowning the signal). Dense tiles (>=8 symbols) duplicated 2x
     — density is the measured failure mode (recall 0.28 on the densest test sheet).
  6. Val split: 4 whole TRAIN sheets held out (sheet-level, no tile leakage).
  7. Pushes tiles (webp, quality 90) + train.jsonl/val.jsonl to a private HF dataset.
"""
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from huggingface_hub import HfApi, hf_hub_download

random.seed(4242)

HF_TOKEN = os.environ["HF_TOKEN"]
DATA_REPO = "timthy45/pnid-extraction-datasets"
OUT_REPO = "timthy45/molmo2-pnid-pointing-data"
TEST_FIXTURES = Path("/Users/tomgeorge/pid-ml/notebooks/stage4/gupta_test_sheets")
WORK = Path("/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6/scratchpad/molmo_ft_prep")
TILE, OVERLAP, UPSCALE = 512, 102, 2.0
EMPTY_FRACTION = 0.25
DENSE_THRESHOLD, DENSE_DUP = 8, 2
VAL_SHEETS = 4

WORK.mkdir(parents=True, exist_ok=True)
(WORK / "tiles").mkdir(exist_ok=True)

# ── 1. fetch + extract Gupta ────────────────────────────────────────────────────
print("downloading Gupta zip from HF...")
z = hf_hub_download(repo_id=DATA_REPO, filename="gupta_pid/PID_Dataset.zip",
                    repo_type="dataset", token=HF_TOKEN)
ex = WORK / "gupta"
if not ex.exists():
    with zipfile.ZipFile(z) as f:
        f.extractall(ex)

# ONLY the raw ground-truth split. The zip is the full research dump — rglob matching
# by stem (first attempt) silently paired sheets with YOLO-run PSEUDOLABELS from
# 2__Stage-1/3__Psuedolabels/ and picked up result charts/crops as "images". Real bug,
# caught by sanity-checking the sample counts before training.
raw = next(ex.rglob("0__raw_data"))
sheets_train = raw / "sheets" / "train"
labels_train = raw / "labels" / "train"
imgs = {p.stem: p for p in sheets_train.iterdir() if p.suffix.lower() in (".jpg", ".png")}
labels = {p.stem: labels_train / f"{p.stem}.txt" for p in sheets_train.iterdir()
          if (labels_train / f"{p.stem}.txt").exists()}
zip_test_stems = {p.stem for p in (raw / "sheets" / "test").iterdir()}
print(f"raw train sheets: {len(imgs)}, labeled: {len(labels)}, zip test split: {len(zip_test_stems)}")

# ── 2. frozen-test leak assert (hard rule) ──────────────────────────────────────
test_stems = {p.stem for p in (TEST_FIXTURES / "sheets").glob("*.jpg")}
assert len(test_stems) == 20, f"expected 20 frozen test stems, got {len(test_stems)}"
assert zip_test_stems == test_stems, (
    f"zip's own test split != frozen fixtures: only_zip={zip_test_stems - test_stems}, "
    f"only_fixtures={test_stems - zip_test_stems}")
train_stems = sorted(set(labels) - test_stems, key=lambda s: int(s) if s.isdigit() else 10**9)
overlap = set(train_stems) & test_stems
assert not overlap, f"TEST LEAK: {overlap}"
print(f"train sheets: {len(train_stems)} (test excluded: {sorted(test_stems)[:5]}...)")

val_stems = set(random.sample(train_stems, VAL_SHEETS))
print(f"val sheets (held out whole): {sorted(val_stems)}")

# ── 3. tile + build samples ─────────────────────────────────────────────────────
def tile_grid(w, h, tile=TILE, overlap=OVERLAP):
    stride = tile - overlap
    out = []
    y0 = 0
    while y0 < h:
        y1 = min(y0 + tile, h)
        x0 = 0
        while x0 < w:
            out.append((x0, y0, min(x0 + tile, w), y1))
            x0 += stride
        y0 += stride
    return out


def enhance(pil):
    return ImageOps.autocontrast(pil.convert("L")).convert("RGB")


samples = {"train": [], "val": []}
empties = {"train": [], "val": []}
stats = Counter()

for stem in train_stems:
    img = Image.open(imgs[stem]).convert("RGB")
    W, H = img.size
    boxes = []
    for line in labels[stem].read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append((cx * W, cy * H))
    split = "val" if stem in val_stems else "train"
    for (x0, y0, x1, y1) in tile_grid(W, H):
        pts = [(px - x0, py - y0) for (px, py) in boxes if x0 <= px < x1 and y0 <= py < y1]
        tw, th = x1 - x0, y1 - y0
        if tw < 64 or th < 64:
            continue
        crop = img.crop((x0, y0, x1, y1))
        up_w, up_h = int(tw * UPSCALE), int(th * UPSCALE)
        crop = enhance(crop.resize((up_w, up_h), Image.LANCZOS))
        # coords in Molmo's 0-1000 space of the tile the model SEES (the upscaled one)
        coords = sorted(
            (round(px * UPSCALE / up_w * 1000), round(py * UPSCALE / up_h * 1000))
            for (px, py) in pts
        )
        rec_id = f"{stem}_{x0}_{y0}"
        rec = {"id": rec_id, "points": coords, "n": len(coords)}
        if coords:
            crop.save(WORK / "tiles" / f"{rec_id}.webp", quality=90)
            samples[split].append(rec)
            stats[f"{split}_symbol_tiles"] += 1
            if len(coords) >= DENSE_THRESHOLD:
                for d in range(DENSE_DUP - 1):
                    samples[split].append({**rec, "id": rec_id, "dup": d + 1})
                stats[f"{split}_dense_dups"] += DENSE_DUP - 1
        else:
            empties[split].append((rec, crop))

# sample empties to the target fraction (save only the kept ones)
for split in ("train", "val"):
    n_keep = int(len(samples[split]) * EMPTY_FRACTION / (1 - EMPTY_FRACTION))
    keep = random.sample(empties[split], min(n_keep, len(empties[split])))
    for rec, crop in keep:
        crop.save(WORK / "tiles" / f"{rec['id']}.webp", quality=90)
        samples[split].append(rec)
        stats[f"{split}_empty_tiles"] += 1
    random.shuffle(samples[split])
    with open(WORK / f"{split}.jsonl", "w") as f:
        for rec in samples[split]:
            f.write(json.dumps(rec) + "\n")

print(dict(stats))
print(f"train samples: {len(samples['train'])}, val samples: {len(samples['val'])}")

# ── 4. push to HF ───────────────────────────────────────────────────────────────
api = HfApi(token=HF_TOKEN)
api.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)
meta = {
    "v": 1, "source": "gupta_pid (class-agnostic), 20 frozen test stems excluded",
    "test_stems": sorted(test_stems), "val_sheets": sorted(val_stems),
    "tile": TILE, "overlap": OVERLAP, "upscale": UPSCALE, "enhance": "autocontrast-gray",
    "coords": "0-1000 ints in the UPSCALED tile's space, (x,y) pairs, sorted",
    "stats": dict(stats),
}
(WORK / "meta.json").write_text(json.dumps(meta, indent=2))
for fn in ("train.jsonl", "val.jsonl", "meta.json"):
    api.upload_file(path_or_fileobj=str(WORK / fn), path_in_repo=fn,
                    repo_id=OUT_REPO, repo_type="dataset")
print("uploading tiles folder (this is the big one)...")
api.upload_folder(folder_path=str(WORK / "tiles"), path_in_repo="tiles",
                  repo_id=OUT_REPO, repo_type="dataset")
print(f"DONE - dataset at {OUT_REPO}")
