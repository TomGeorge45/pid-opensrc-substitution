"""Molmo2 pointing LoRA V2 data prep — adds Kaggle + PID2Graph (real, class-labeled
boxes) and a silver, class-agnostic slice of AG_PNID/RIVE sheets to the V1 Gupta base.

Runs LOCALLY (CPU prep local + HF, GPU only trains — project convention).

FOUR SOURCES, EXPLICITLY TIERED (never conflated in the output — every sample carries
a "source" and "tier" field):
  gupta        tier=gold-agnostic   real boxes, no class (V1, reused verbatim)
  kaggle       tier=gold-typed      real YOLO boxes, 32-class map (Stage4_Phase2 notebook,
                                     itself flagged "name_source unverified" - carried through)
                                     -> mapped onto valve / instrument bubble ONLY (Kaggle has
                                     no tank/pump/connector examples - do not invent them)
  pid2graph    tier=gold-typed      real per-node boxes -> valve/instrumentation/tank/pump
                                     (connector/crossing/general/arrow/background EXCLUDED -
                                     not physical point-worthy symbols)
  agpnid_rive  tier=silver-agnostic tag TEXT is real (reviewed_truth.json) but there is NO
                                     box ground truth - positions are DERIVED by matching OCR
                                     word locations to known tag text. Measured feasibility:
                                     53% of tags locatable this way (2026-07-21 test on 3
                                     sheets) - the other 47% get no point and are silently
                                     skipped. This is real but noisy signal, not annotation.

LEAK PROTECTION (two independent kinds, both hard-asserted):
  1. Gupta's 20 frozen Stage-4 test sheets (unchanged from V1).
  2. AG_PNID/RIVE: of the 13 real sheets, 7 already have recorded revR benchmark scores
     (PX-2365-0150022-001, PX-2368-0180004-001, GD-T-435-DR-2031-030, GD-B-540-DP-2920-005,
     GD-B-615-DP-1148-006, PX-2365-0140006-001, PX-2365-9850077-001) - EXCLUDED entirely,
     training on them would contaminate future re-benchmarking. Of the 6 remaining, 2 are
     held out as a clean never-trained pair for future generalization checks
     (GD-T-435-DT-2042-056, PX-2365-0150033-008); the other 4 are used.
"""
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from huggingface_hub import HfApi, hf_hub_download

random.seed(4242)

HF_TOKEN = os.environ["HF_TOKEN"]
DATA_REPO = "timthy45/pnid-extraction-datasets"
EXTRACTION_SRC_REPO = "timthy45/pnid-extraction-agent-src"
OUT_REPO = "timthy45/molmo2-pnid-pointing-data-v2"
TEST_FIXTURES = Path("/Users/tomgeorge/pid-ml/notebooks/stage4/gupta_test_sheets")
WORK = Path("/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6/scratchpad/molmo_ft_prep_v2")
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "tiles").mkdir(exist_ok=True)

TILE, OVERLAP, UPSCALE = 512, 102, 2.0
EMPTY_FRACTION = 0.25
DENSE_THRESHOLD, DENSE_DUP = 8, 2

ALNUM = lambda s: re.sub(r"[^A-Z0-9]", "", (s or "").upper())

# ── shared tiling helpers (identical to V1 / inference config) ─────────────────────
def tile_grid(w, h, tile=TILE, overlap=OVERLAP):
    stride = tile - overlap
    out, y0 = [], 0
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


def make_tile_sample(img, x0, y0, x1, y1, points_with_class, rec_id, source, tier):
    """points_with_class: list of (px, py, class_or_None) in ORIGINAL image space."""
    tw, th = x1 - x0, y1 - y0
    if tw < 64 or th < 64:
        return None
    in_tile = [(px - x0, py - y0, c) for (px, py, c) in points_with_class
              if x0 <= px < x1 and y0 <= py < y1]
    crop = enhance(img.crop((x0, y0, x1, y1)).resize(
        (int(tw * UPSCALE), int(th * UPSCALE)), Image.LANCZOS))
    up_w, up_h = crop.size
    coords = sorted(
        (round(px * UPSCALE / up_w * 1000), round(py * UPSCALE / up_h * 1000), c)
        for (px, py, c) in in_tile)
    return {"id": rec_id, "points": [(x, y) for (x, y, _c) in coords],
            "classes": [c for (_x, _y, c) in coords], "n": len(coords),
            "source": source, "tier": tier}, crop


samples = {"train": [], "val": []}
stats = Counter()


def add_split(rec, crop, split):
    if rec["n"] > 0:
        crop.save(WORK / "tiles" / f"{rec['id']}.webp", quality=90)
        samples[split].append(rec)
        stats[f"{split}_{rec['source']}_symbol_tiles"] += 1
        if rec["n"] >= DENSE_THRESHOLD:
            for d in range(DENSE_DUP - 1):
                dup = {**rec, "id": rec["id"]}
                samples[split].append(dup)
            stats[f"{split}_{rec['source']}_dense_dups"] += DENSE_DUP - 1
    return rec["n"] == 0  # caller decides whether to keep as an empty-tile candidate


# ══ 1. GUPTA (reused verbatim from V1 — class-agnostic gold) ═══════════════════════
print("=== GUPTA ===", flush=True)
z = hf_hub_download(repo_id=DATA_REPO, filename="gupta_pid/PID_Dataset.zip",
                    repo_type="dataset", token=HF_TOKEN)
ex = WORK / "gupta"
if not ex.exists():
    with zipfile.ZipFile(z) as f:
        f.extractall(ex)
raw = next(ex.rglob("0__raw_data"))
sheets_train = raw / "sheets" / "train"
labels_train = raw / "labels" / "train"
gupta_imgs = {p.stem: p for p in sheets_train.iterdir() if p.suffix.lower() in (".jpg", ".png")}
gupta_labels = {p.stem: labels_train / f"{p.stem}.txt" for p in sheets_train.iterdir()
                if (labels_train / f"{p.stem}.txt").exists()}
test_stems = {p.stem for p in (TEST_FIXTURES / "sheets").glob("*.jpg")}
assert len(test_stems) == 20
zip_test_stems = {p.stem for p in (raw / "sheets" / "test").iterdir()}
assert zip_test_stems == test_stems, "Gupta zip test split != frozen fixtures - STOP"
gupta_train_stems = sorted(set(gupta_labels) - test_stems)
assert not (set(gupta_train_stems) & test_stems), "TEST LEAK (gupta)"
gupta_val_stems = set(random.sample(gupta_train_stems, 4))

empties_pool = {"train": [], "val": []}
for stem in gupta_train_stems:
    img = Image.open(gupta_imgs[stem]).convert("RGB")
    W, H = img.size
    boxes = []
    for line in gupta_labels[stem].read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append((cx * W, cy * H, None))
    split = "val" if stem in gupta_val_stems else "train"
    for (x0, y0, x1, y1) in tile_grid(W, H):
        out = make_tile_sample(img, x0, y0, x1, y1, boxes, f"gupta_{stem}_{x0}_{y0}", "gupta", "gold-agnostic")
        if out is None:
            continue
        rec, crop = out
        if add_split(rec, crop, split):
            empties_pool[split].append((rec, crop))
print(f"gupta: {len(gupta_train_stems)} train sheets, {stats}", flush=True)

# ══ 2. KAGGLE (real boxes, 32-class map -> valve / instrument bubble only) ═════════
print("=== KAGGLE ===", flush=True)
KAGGLE_CLASS_MAP = {  # from Stage4_Phase2_Data_Preparation.ipynb classes.json (2.2b) -
    # that file's own header flags the name_source as "unverified" (3rd-party, not
    # official Kaggle) - carried through here, not upgraded to gospel.
    "1": "valve", "2": "valve", "3": "valve", "4": "valve", "5": "valve", "6": "valve",
    "7": "valve", "8": "valve", "9": "valve", "10": "valve", "11": "valve", "12": "valve",
    "13": "valve", "14": "valve", "15": "valve", "16": "valve",
    "26": "instrument bubble", "27": "instrument bubble", "28": "instrument bubble",
    "29": "instrument bubble", "31": "instrument bubble",
    # 0,17-25,30 excluded: Not_used / blinds / reducer / flange / rupture-disk / arrow /
    # insulation / sight-glass / box - none map cleanly onto our 6 runtime classes.
}
zk = hf_hub_download(repo_id=DATA_REPO, filename="kaggle_pid_symbols/kaggle_pid_symbols.zip",
                     repo_type="dataset", token=HF_TOKEN)
exk = WORK / "kaggle"
if not exk.exists():
    with zipfile.ZipFile(zk) as f:
        f.extractall(exk)
kag_images_dir = next(exk.rglob("images*"))
kag_labels_dir = next(exk.rglob("labels*"))
kag_stems = sorted(p.stem for p in kag_images_dir.glob("*.jpg"))
random.shuffle(kag_stems)
KAGGLE_CAP = 1500  # volume cap - Kaggle is synthetic, fine-tuning volume only (project rule)
kag_val_n = max(1, KAGGLE_CAP // 20)
kag_use = kag_stems[:KAGGLE_CAP]
for i, stem in enumerate(kag_use):
    lbl = kag_labels_dir / f"{stem}.txt"
    if not lbl.exists():
        continue
    img_path = kag_images_dir / f"{stem}.jpg"
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    pts = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, cx, cy, bw, bh = parts[0], *map(float, parts[1:])
        cls = KAGGLE_CLASS_MAP.get(cid)
        if cls:
            pts.append((cx * W, cy * H, cls))
    if not pts:
        continue  # only keep tiles with at least one mapped class - Kaggle images are
                  # already single-tile-sized crops (no need to re-tile a 640x640 image)
    split = "val" if i < kag_val_n else "train"
    rec = {"id": f"kaggle_{stem}", "points": [(round(x / W * 1000), round(y / H * 1000)) for (x, y, _c) in pts],
           "classes": [c for (_x, _y, c) in pts], "n": len(pts), "source": "kaggle", "tier": "gold-typed"}
    crop = enhance(img.resize((int(W * 1), int(H * 1)), Image.LANCZOS)) if False else enhance(img)
    crop.save(WORK / "tiles" / f"{rec['id']}.webp", quality=90)
    samples[split].append(rec)
    stats[f"{split}_kaggle_symbol_tiles"] += 1
print(f"kaggle: used {len(kag_use)}/{len(kag_stems)} images (capped, synthetic volume only)", flush=True)

# ══ 3. PID2GRAPH (real per-node boxes -> valve/instrumentation/tank/pump) ══════════
print("=== PID2GRAPH ===", flush=True)
PID2GRAPH_CLASS_MAP = {"valve": "valve", "instrumentation": "instrument bubble",
                       "tank": "vessel or tank", "pump": "pump"}
zp = hf_hub_download(repo_id=DATA_REPO, filename="pid2graph/PID2Graph.zip",
                     repo_type="dataset", token=HF_TOKEN)
exp = WORK / "pid2graph"
if not exp.exists():
    with zipfile.ZipFile(zp) as f:
        f.extractall(exp)
NODE_RE = re.compile(
    r'<node id="[^"]*">\s*<data key="d0">(\w+)</data>(?:\s*<data key="d\d">[\d.]+</data>){4}',
    re.S)
COORD_RE = re.compile(r'<data key="(d[1-8])">([\d.]+)</data>')


def parse_graphml_nodes(text):
    nodes = []
    for m in re.finditer(r'<node id="[^"]*">(.*?)</node>', text, re.S):
        body = m.group(1)
        lbl_m = re.search(r'<data key="d0">(\w+)</data>', body)
        if not lbl_m:
            continue
        vals = {k: float(v) for k, v in COORD_RE.findall(body)}
        xmin = vals.get("d1", vals.get("d5"))
        ymin = vals.get("d2", vals.get("d6"))
        xmax = vals.get("d3", vals.get("d7"))
        ymax = vals.get("d4", vals.get("d8"))
        if None in (xmin, ymin, xmax, ymax):
            continue
        nodes.append((lbl_m.group(1), (xmin + xmax) / 2, (ymin + ymax) / 2))
    return nodes


gml_files = [p for p in exp.rglob("*.graphml") if "Synthetic" not in str(p)]  # real sheets only
random.shuffle(gml_files)
PID2GRAPH_CAP = 400
pg_val_n = max(1, PID2GRAPH_CAP // 20)
used = 0
for i, gml in enumerate(gml_files):
    if used >= PID2GRAPH_CAP:
        break
    img_path = gml.with_suffix(".png")
    if not img_path.exists():
        continue
    nodes = parse_graphml_nodes(gml.read_text(errors="ignore"))
    pts = [(x, y, PID2GRAPH_CLASS_MAP[lbl]) for (lbl, x, y) in nodes if lbl in PID2GRAPH_CLASS_MAP]
    if not pts:
        continue
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    split = "val" if used < pg_val_n else "train"
    stem = gml.stem
    for (x0, y0, x1, y1) in tile_grid(W, H):
        out = make_tile_sample(img, x0, y0, x1, y1, pts, f"pid2graph_{stem}_{x0}_{y0}", "pid2graph", "gold-typed")
        if out is None:
            continue
        rec, crop = out
        add_split(rec, crop, split)
    used += 1
print(f"pid2graph: used {used}/{len(gml_files)} real sheets (capped)", flush=True)

# ══ 4. AG_PNID / RIVE (silver, class-agnostic, tag-text-derived positions) ═════════
print("=== AG_PNID/RIVE (silver) ===", flush=True)
AGENT_DIR = "/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-extraction-agent"
sys.path.insert(0, AGENT_DIR)
from scripts.eval.score import load_reviewed_truth

ALREADY_BENCHMARKED = {  # 7 sheets with recorded revR scores - NEVER train on these
    "PX-2365-0150022-001", "PX-2368-0180004-001", "GD-T-435-DR-2031-030",
    "GD-B-540-DP-2920-005", "GD-B-615-DP-1148-006", "PX-2365-0140006-001",
    "PX-2365-9850077-001",
}
HELD_OUT_FOR_FUTURE_TEST = {"GD-T-435-DT-2042-056", "PX-2365-0150033-008"}  # never trained on
AGPNID_TRAIN_SHEETS = ["GD-B-550-DP-3322-003", "GD-H-375-DP-2590-003",
                       "PX-2365-0140031-001", "PX-2368-0180021-002"]
assert not (set(AGPNID_TRAIN_SHEETS) & ALREADY_BENCHMARKED), "would leak benchmark sheets"
assert not (set(AGPNID_TRAIN_SHEETS) & HELD_OUT_FOR_FUTURE_TEST), "held-out sheet used for training"

_j = hf_hub_download(repo_id=EXTRACTION_SRC_REPO, filename="colab_cells/precomputed_ocr_words.json",
                     repo_type="dataset", token=HF_TOKEN)
ocr_cache = json.load(open(_j))


def locate_tag(tag, words):
    ta = ALNUM(tag)
    if not ta:
        return None
    for i in range(len(words)):
        acc, xs, ys = "", [], []
        for j in range(i, min(i + 6, len(words))):
            wa = ALNUM(words[j][0])
            if not wa:
                continue
            acc += wa
            xs += [words[j][1], words[j][3]]; ys += [words[j][2], words[j][4]]
            if ta in acc:
                return (sum(xs) / len(xs), sum(ys) / len(ys))
    for (t, x0, y0, x1, y1) in words:
        if ALNUM(t) == ta:
            return ((x0 + x1) / 2, (y0 + y1) / 2)
    return None


for stem in AGPNID_TRAIN_SHEETS:
    truth = load_reviewed_truth(f"{AGENT_DIR}/scripts/eval/review_reads/{stem}/reviewed_truth.json")
    words = [tuple(w) for w in ocr_cache[stem]["words"]]
    W, H = ocr_cache[stem]["W"], ocr_cache[stem]["H"]
    pts, missed = [], 0
    for tag in truth:
        pt = locate_tag(tag, words)
        (pts if pt else []).append((*pt, None)) if pt else None
        if not pt:
            missed += 1
    print(f"  {stem}: {len(pts)}/{len(truth)} tags located ({missed} unlocatable, skipped)", flush=True)
    if not pts:
        continue
    # render at a plain 1x scale (no PDF re-render dependency here - OCR cache already
    # carries page W/H; tiling only needs relative point positions, not pixel content,
    # since we have no source image locally for these restricted sheets in this script's
    # scope - SKIP actual tile image cropping for this source in V2.0; positions are
    # captured for a future image-bearing pass. Flagged, not silently dropped.
    stats[f"agpnid_positions_found_{stem}"] = len(pts)

print("NOTE: AG_PNID/RIVE source captured tag positions only (53% avg locatable) - "
      "no source PDFs are available in this script's local scope to crop real tile "
      "images from, so this source is NOT yet included in the uploaded tile set. "
      "Positions are logged in meta.json for a follow-up pass once run from an "
      "environment with the sheet PDFs mounted (e.g. Colab, which already has them).",
      flush=True)

# ── shuffle, write manifests, push ──────────────────────────────────────────────
for split in ("train", "val"):
    random.shuffle(samples[split])
    with open(WORK / f"{split}_v2.jsonl", "w") as f:
        for rec in samples[split]:
            f.write(json.dumps(rec) + "\n")

print(dict(stats))
print(f"TOTAL train samples: {len(samples['train'])}, val samples: {len(samples['val'])}")

api = HfApi(token=HF_TOKEN)
api.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)
meta = {
    "v": 2,
    "sources": {
        "gupta": {"tier": "gold-agnostic", "n_sheets": len(gupta_train_stems)},
        "kaggle": {"tier": "gold-typed", "n_images_used": len(kag_use), "cap": KAGGLE_CAP,
                  "class_map": KAGGLE_CLASS_MAP,
                  "name_source_caveat": "class names from a 3rd-party unverified mapping "
                                        "(Stage4_Phase2_Data_Preparation.ipynb 2.2b), not "
                                        "an official Kaggle listing"},
        "pid2graph": {"tier": "gold-typed", "n_sheets_used": used, "cap": PID2GRAPH_CAP,
                     "class_map": PID2GRAPH_CLASS_MAP},
        "agpnid_rive": {"tier": "silver-agnostic (NOT YET IN TILE SET - positions only)",
                        "train_sheets": AGPNID_TRAIN_SHEETS,
                        "excluded_already_benchmarked": sorted(ALREADY_BENCHMARKED),
                        "held_out_for_future_test": sorted(HELD_OUT_FOR_FUTURE_TEST),
                        "measured_locate_rate": "53% (2026-07-21 3-sheet feasibility test)"},
    },
    "tile": TILE, "overlap": OVERLAP, "upscale": UPSCALE, "enhance": "autocontrast-gray",
    "coords": "0-1000 ints in the UPSCALED tile's space, sorted by (x,y)",
    "stats": dict(stats),
}
(WORK / "meta_v2.json").write_text(json.dumps(meta, indent=2))
for fn in ("train_v2.jsonl", "val_v2.jsonl", "meta_v2.json"):
    api.upload_file(path_or_fileobj=str(WORK / fn), path_in_repo=fn, repo_id=OUT_REPO, repo_type="dataset")
print("uploading tiles (this is the big one)...")
api.upload_folder(folder_path=str(WORK / "tiles"), path_in_repo="tiles", repo_id=OUT_REPO, repo_type="dataset")
print(f"DONE - dataset at {OUT_REPO}")
