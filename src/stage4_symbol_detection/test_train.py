"""Tests for the torch-free parts of train.py: example building (tiling Gupta sheets,
cropping, target-text formatting). The actual training loop (build_labeled_inputs, setup_lora,
time_training_steps) needs torch/transformers/peft and can only be run in Colab — see
Stage4_Checklist_Status.md for the required manual timing-probe step. Run with:
    python3 -m src.stage4_symbol_detection.test_train
"""

import json
import tempfile
from pathlib import Path

from . import finetune_dataset, train
from .molmo_candidate import parse as parse_molmo
from .qwen_candidate import parse as parse_qwen


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label


def _make_fake_repo(tmp_root):
    """A Gupta sheet just over 1 tile wide/tall (forces >1 tile) + one Kaggle tile."""
    tmp_root = Path(tmp_root)
    gupta_raw = tmp_root / "gupta_pid" / "PID_Dataset" / "0__raw_data"
    kaggle_root = tmp_root / "kaggle_pid_symbols"
    for split in ("train", "test"):
        (gupta_raw / "sheets" / split).mkdir(parents=True, exist_ok=True)
        (gupta_raw / "labels" / split).mkdir(parents=True, exist_ok=True)
    (kaggle_root / "images").mkdir(parents=True, exist_ok=True)
    (kaggle_root / "labels").mkdir(parents=True, exist_ok=True)

    from PIL import Image

    # 1200x1200 sheet -> 4 tiles at 1024/205 (2x2 grid), with one box in the top-left tile
    sheet_w, sheet_h = 1200, 1200
    Image.new("RGB", (sheet_w, sheet_h), "white").save(gupta_raw / "sheets" / "train" / "s1.jpg")
    # box centered at (100, 100), small — only in top-left tile
    (gupta_raw / "labels" / "train" / "s1.txt").write_text(
        f"0 {100/sheet_w} {100/sheet_h} {20/sheet_w} {20/sheet_h}\n"
    )
    (tmp_root / "test_ids.json").write_text(json.dumps({"test_ids": []}))

    Image.new("RGB", (1280, 1280), "white").save(kaggle_root / "images" / "k1.jpg")
    (kaggle_root / "labels" / "k1.txt").write_text("1 0.5 0.5 0.1 0.1\n")

    return tmp_root


def _patch_paths(tmp_root):
    finetune_dataset.ROOT = tmp_root
    finetune_dataset.gupta_p = tmp_root / "gupta_pid"
    finetune_dataset.kaggle_p = tmp_root / "kaggle_pid_symbols"
    train.assemble_manifest = finetune_dataset.assemble_manifest


def test_gupta_sheet_gets_tiled_not_used_whole():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = _make_fake_repo(tmp)
        _patch_paths(tmp_root)

        manifest = finetune_dataset.assemble_manifest(expected_gupta_train_count=None)
        examples = train.build_examples("qwen3vl", manifest=manifest)

        gupta_examples = [e for e in examples if e["source"] == "gupta"]
        check(f"1200x1200 sheet produces >1 tile example (got {len(gupta_examples)})", len(gupta_examples) > 1)
        check("each gupta tile example has a crop region", all(e["crop"] is not None for e in gupta_examples))

        kaggle_examples = [e for e in examples if e["source"] == "kaggle"]
        check("kaggle examples have no crop (already tile-sized)", all(e["crop"] is None for e in kaggle_examples))


def test_gupta_box_only_appears_in_intersecting_tile_targets():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = _make_fake_repo(tmp)
        _patch_paths(tmp_root)

        manifest = finetune_dataset.assemble_manifest(expected_gupta_train_count=None)
        examples = train.build_examples("qwen3vl", manifest=manifest)
        gupta_examples = [e for e in examples if e["source"] == "gupta"]

        with_box = [e for e in gupta_examples if e["n_boxes"] > 0]
        check(f"exactly one tile contains the single box (got {len(with_box)})", len(with_box) == 1)

        detections, error = parse_qwen(with_box[0]["target_text"])
        check(f"gupta tile target parses via the real qwen parser (error={error})", error is None)
        check("gupta box entity_type is generic 'symbol' (class-agnostic)",
              detections[0]["entity_type"] == "symbol")


def test_molmo_candidate_uses_points_format():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = _make_fake_repo(tmp)
        _patch_paths(tmp_root)

        manifest = finetune_dataset.assemble_manifest(expected_gupta_train_count=None)
        examples = train.build_examples("molmo2", manifest=manifest)
        gupta_examples = [e for e in examples if e["source"] == "gupta" and e["n_boxes"] > 0]

        check("molmo2 target uses <points> tag, not JSON array",
              gupta_examples[0]["target_text"].startswith("<points"))

        tile_w = gupta_examples[0]["crop"][2] - gupta_examples[0]["crop"][0]
        tile_h = gupta_examples[0]["crop"][3] - gupta_examples[0]["crop"][1]
        detections, error = parse_molmo(gupta_examples[0]["target_text"], tile_w, tile_h)
        check(f"molmo2 gupta tile target parses via the real molmo parser (error={error})", error is None)


if __name__ == "__main__":
    test_gupta_sheet_gets_tiled_not_used_whole()
    test_gupta_box_only_appears_in_intersecting_tile_targets()
    test_molmo_candidate_uses_points_format()
    print("\nAll train.py example-building tests passed.")
