"""Dev-set parse-rate check (checklist 2.5 confirm bar): each candidate's parser should
round-trip >=95% of a small labeled dev set without manual fixup. Measures parse success,
not detection accuracy — a model can parse perfectly while still being wrong about what
it sees.
"""

import random

from .data_utils import kaggle_p


def build_dev_set(n=10, pool_size=200, seed=0):
    """Sample n labeled Kaggle tiles for a parse-rate check."""
    random.seed(seed)
    pool = []
    for lbl in (kaggle_p / "labels").glob("*.txt"):
        n_boxes = len([l for l in lbl.read_text().splitlines() if l.strip()])
        if n_boxes >= 1:
            pool.append((kaggle_p / "images" / f"{lbl.stem}.jpg", lbl, n_boxes))
        if len(pool) >= pool_size:
            break
    return random.sample(pool, n)


def run_parse_check(dev_set, run_fn, parse_fn, needs_dims=False):
    """run_fn(image) -> (raw_text, latency). parse_fn(text[, w, h]) -> (detections, error).
    Set needs_dims=True for parsers that take image width/height (e.g. Molmo's normalized
    coords). Returns (results, parse_rate)."""
    from PIL import Image

    results = []
    for img_path, lbl_path, gt_n in dev_set:
        img = Image.open(img_path).convert("RGB")
        raw_text, latency = run_fn(img)
        if needs_dims:
            detections, error = parse_fn(raw_text, img.width, img.height)
        else:
            detections, error = parse_fn(raw_text)
        results.append({
            "image": img_path.name, "gt_boxes": gt_n,
            "parsed_ok": error is None,
            "n_detections": len(detections) if detections else 0,
            "error": error, "latency": latency,
        })
        status = "OK" if error is None else f"FAIL: {error}"
        print(f"{img_path.name:20s} gt={gt_n:2d} pred={len(detections) if detections else 0:2d} "
              f"latency={latency:.2f}s  [{status}]")

    n_ok = sum(r["parsed_ok"] for r in results)
    parse_rate = n_ok / len(results)
    print(f"\nParse success: {n_ok}/{len(results)} ({parse_rate*100:.0f}%)")
    print("✓ meets ≥95% bar" if parse_rate >= 0.95 else "✗ BELOW 95% bar")

    avg_latency = sum(r["latency"] for r in results) / len(results)
    print(f"Avg latency: {avg_latency:.2f}s/tile")
    return results, parse_rate
