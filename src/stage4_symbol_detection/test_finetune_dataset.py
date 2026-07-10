"""Confirm-bar test for Phase 5.1 — the single most important check in the whole
benchmarking checklist: a train/test leak must be caught, never silently pass. Run with:
    python3 -m src.stage4_symbol_detection.test_finetune_dataset
No GPU/Colab/real data needed — builds a synthetic directory structure in a temp dir.
"""

import json
import shutil
import tempfile
from pathlib import Path

from . import data_utils, finetune_dataset


def _make_fake_repo(tmp_root, leak=False):
    """Builds a minimal fake Drive layout: gupta (sheets/labels train+test) + kaggle
    (images+labels) + test_ids.json. If leak=True, deliberately puts a "test" sheet ID
    into the train folder too, to verify assert_no_leak actually catches it."""
    tmp_root = Path(tmp_root)
    gupta_raw = tmp_root / "gupta_pid" / "PID_Dataset" / "0__raw_data"
    kaggle_root = tmp_root / "kaggle_pid_symbols"

    for split in ("train", "test"):
        (gupta_raw / "sheets" / split).mkdir(parents=True, exist_ok=True)
        (gupta_raw / "labels" / split).mkdir(parents=True, exist_ok=True)
    (kaggle_root / "images").mkdir(parents=True, exist_ok=True)
    (kaggle_root / "labels").mkdir(parents=True, exist_ok=True)

    train_ids = [f"train_{i}" for i in range(3)]
    test_ids = [f"test_{i}" for i in range(2)]

    for sid in train_ids:
        (gupta_raw / "sheets" / "train" / f"{sid}.jpg").write_bytes(b"fake")
        (gupta_raw / "labels" / "train" / f"{sid}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    for sid in test_ids:
        (gupta_raw / "sheets" / "test" / f"{sid}.jpg").write_bytes(b"fake")
        (gupta_raw / "labels" / "test" / f"{sid}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    if leak:
        # deliberately duplicate a "test" sheet ID into the train folder too
        leaked_id = test_ids[0]
        (gupta_raw / "sheets" / "train" / f"{leaked_id}.jpg").write_bytes(b"fake")
        (gupta_raw / "labels" / "train" / f"{leaked_id}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    for i in range(4):
        (kaggle_root / "images" / f"k_{i}.jpg").write_bytes(b"fake")
        (kaggle_root / "labels" / f"k_{i}.txt").write_text("1 0.5 0.5 0.1 0.1\n")

    (tmp_root / "test_ids.json").write_text(json.dumps({"test_ids": test_ids}))

    return tmp_root


def _patch_paths(tmp_root):
    """Monkeypatch data_utils' module-level paths to point at the fake repo, and re-point
    finetune_dataset's already-imported references to match (both modules imported the
    names directly, so both need patching)."""
    data_utils.ROOT = tmp_root
    data_utils.gupta_p = tmp_root / "gupta_pid"
    data_utils.kaggle_p = tmp_root / "kaggle_pid_symbols"
    finetune_dataset.ROOT = tmp_root
    finetune_dataset.gupta_p = tmp_root / "gupta_pid"
    finetune_dataset.kaggle_p = tmp_root / "kaggle_pid_symbols"


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label


def test_clean_assembly_succeeds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = _make_fake_repo(tmp, leak=False)
        _patch_paths(tmp_root)

        manifest = finetune_dataset.assemble_manifest(expected_gupta_train_count=None)
        check(f"3 train sheets collected (got {manifest['counts']['gupta_train_sheets']})",
              manifest["counts"]["gupta_train_sheets"] == 3)
        check(f"4 kaggle images collected (got {manifest['counts']['kaggle_pretrain_images']})",
              manifest["counts"]["kaggle_pretrain_images"] == 4)
        check("no test sheet IDs present in gupta_train",
              not (set(i["id"] for i in manifest["gupta_train"]) & set(manifest["test_ids_excluded"])))


def test_leak_is_caught_not_silently_passed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = _make_fake_repo(tmp, leak=True)
        _patch_paths(tmp_root)

        raised = False
        try:
            finetune_dataset.assemble_manifest(expected_gupta_train_count=None)
        except AssertionError as e:
            raised = True
            check("leak error message names the offending sheet ID", "test_0" in str(e))
        check("assemble_manifest raises AssertionError on a real leak (not silent)", raised)


def test_write_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = _make_fake_repo(tmp, leak=False)
        _patch_paths(tmp_root)

        manifest = finetune_dataset.assemble_manifest(expected_gupta_train_count=None)
        out_path = finetune_dataset.write_manifest(manifest, out_path=tmp_root / "manifest.json")
        reloaded = json.loads(out_path.read_text())
        check("written manifest round-trips correctly", reloaded == manifest)


if __name__ == "__main__":
    test_clean_assembly_succeeds()
    test_leak_is_caught_not_silently_passed()
    test_write_manifest_roundtrip()
    print("\nAll Phase 5.1 confirm-bar tests passed.")
