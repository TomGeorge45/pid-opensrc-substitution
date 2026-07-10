"""Phase 5.1 — assemble the domain-adaptation fine-tuning dataset.

Per spec §5 layering rule: domain-adaptation base is task-neutral (pretrain on Kaggle volume
+ Gupta train sheets); detection is a separate LoRA on top. This module assembles the
*dataset manifest* — model-specific training format conversion happens per-candidate, since
the 2026-07-10 decision fine-tunes all 3 (Qwen3-VL, InternVL3, Molmo2-O-7B), not just a
zero-shot winner.

The single most important confirmation in the whole benchmarking checklist (5.1's own words):
zero overlap between the assembled training set and the frozen test_ids.json. A leak here
invalidates every downstream result, so `assert_no_leak` is not optional and is called
automatically by `assemble_manifest` — there is no code path that skips it.
"""

import json

from .data_utils import ROOT, gupta_p, kaggle_p


def load_test_ids():
    """Returns the frozen 20 test-sheet IDs from test_ids.json (checklist 2.1)."""
    test_ids_path = ROOT / "test_ids.json"
    return set(json.loads(test_ids_path.read_text())["test_ids"])


def collect_gupta_train_sheets():
    """Returns list of (sheet_id, image_path, label_path) for Gupta's 72 train sheets."""
    raw = gupta_p / "PID_Dataset" / "0__raw_data"
    items = []
    for lbl in (raw / "labels" / "train").glob("*.txt"):
        img_candidates = list((raw / "sheets" / "train").glob(f"{lbl.stem}.*"))
        if img_candidates:
            items.append((lbl.stem, img_candidates[0], lbl))
    return items


def collect_kaggle_pretrain_images():
    """Returns list of (image_id, image_path, label_path) for all Kaggle tiles — pretrain
    volume, not subject to the Gupta train/test split (Kaggle has no test role here; it's
    synthetic typing/volume data only, per CLAUDE.md hard rule 4)."""
    items = []
    for lbl in (kaggle_p / "labels").glob("*.txt"):
        img_path = kaggle_p / "images" / f"{lbl.stem}.jpg"
        if img_path.exists():
            items.append((lbl.stem, img_path, lbl))
    return items


def assert_no_leak(gupta_train_ids, test_ids):
    """The critical check. Raises AssertionError with the offending IDs if any overlap —
    never silently continues on failure."""
    overlap = set(gupta_train_ids) & set(test_ids)
    assert not overlap, f"TRAIN/TEST LEAK DETECTED — {len(overlap)} sheet(s) in both: {sorted(overlap)}"


def assemble_manifest(data_version="data-v1", expected_gupta_train_count=72):
    """Builds and returns the fine-tuning dataset manifest. Asserts zero train/test leak
    before returning anything — this call cannot succeed with a leak present.

    expected_gupta_train_count: sanity-checked against the real 72-sheet split (checklist
    1.2) by default; pass None to skip (e.g. for tests against synthetic data).

    Returns a dict: {"data_version", "gupta_train": [...], "kaggle_pretrain": [...],
    "counts": {...}}, where each list entry is {"id", "image_path", "label_path", "source"}.
    Model-agnostic — the 2026-07-10 decision fine-tunes all 3 candidates from this same
    manifest, converting to each one's training format separately.
    """
    test_ids = load_test_ids()
    gupta_train = collect_gupta_train_sheets()
    gupta_train_ids = [item[0] for item in gupta_train]

    assert_no_leak(gupta_train_ids, test_ids)
    if expected_gupta_train_count is not None:
        assert len(gupta_train) == expected_gupta_train_count, (
            f"expected {expected_gupta_train_count} Gupta train sheets, got {len(gupta_train)}"
        )

    kaggle_items = collect_kaggle_pretrain_images()

    manifest = {
        "data_version": data_version,
        "test_ids_excluded": sorted(test_ids),
        "gupta_train": [
            {"id": sid, "image_path": str(img), "label_path": str(lbl), "source": "gupta"}
            for sid, img, lbl in gupta_train
        ],
        "kaggle_pretrain": [
            {"id": iid, "image_path": str(img), "label_path": str(lbl), "source": "kaggle"}
            for iid, img, lbl in kaggle_items
        ],
        "counts": {
            "gupta_train_sheets": len(gupta_train),
            "kaggle_pretrain_images": len(kaggle_items),
        },
    }
    return manifest


def write_manifest(manifest, out_path=None):
    """Writes the manifest to Drive (default: ROOT/finetune_manifest.json)."""
    if out_path is None:
        out_path = ROOT / "finetune_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path
