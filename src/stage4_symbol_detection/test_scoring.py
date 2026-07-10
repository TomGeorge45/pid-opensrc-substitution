"""Confirm-bar tests for Phase 3 (checklist 3.1, 3.2, 3.3). Run directly with:
    python3 -m src.stage4_symbol_detection.test_scoring
No GPU/Colab needed — pure logic tests.
"""

from .scoring import (
    average_precision,
    match_predictions_to_gt,
    point_in_box,
    precision_recall_f1,
    typing_accuracy_gt_crop,
)
from .stitch import stitch_and_dedup


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label


def test_point_in_box():
    box = [10, 10, 50, 50]
    check("point clearly inside GT box -> hit", point_in_box((30, 30), box))
    check("point clearly outside GT box -> miss", not point_in_box((100, 100), box))
    check("point on boundary -> hit (inclusive)", point_in_box((10, 10), box))


def test_precision_recall_f1_confirm_bar():
    gt_boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]

    perfect_preds = [
        {"bbox": [1, 1, 9, 9], "confidence": 0.9, "entity_type": "valve"},
        {"bbox": [21, 21, 29, 29], "confidence": 0.9, "entity_type": "valve"},
        {"bbox": [41, 41, 49, 49], "confidence": 0.9, "entity_type": "valve"},
    ]
    result = precision_recall_f1(perfect_preds, gt_boxes)
    check(f"perfect prediction -> precision=1.0 (got {result['precision']})", result["precision"] == 1.0)
    check(f"perfect prediction -> recall=1.0 (got {result['recall']})", result["recall"] == 1.0)
    check(f"perfect prediction -> f1=1.0 (got {result['f1']})", result["f1"] == 1.0)

    empty_result = precision_recall_f1([], gt_boxes)
    check(f"empty prediction -> recall=0.0 (got {empty_result['recall']})", empty_result["recall"] == 0.0)
    check(f"empty prediction -> f1=0.0 (got {empty_result['f1']})", empty_result["f1"] == 0.0)

    half_preds = [perfect_preds[0], perfect_preds[1]]  # 2 of 3 GT boxes found, no false positives
    half_result = precision_recall_f1(half_preds, gt_boxes)
    check(
        f"half-correct -> recall in (0,1) sane middle (got {half_result['recall']})",
        0.0 < half_result["recall"] < 1.0,
    )
    check(f"half-correct -> recall == 2/3 exactly (got {half_result['recall']})", abs(half_result["recall"] - 2 / 3) < 1e-9)
    check(f"half-correct -> precision == 1.0, no false positives (got {half_result['precision']})", half_result["precision"] == 1.0)


def test_average_precision_none_when_no_confidence():
    gt_boxes = [[0, 0, 10, 10]]
    no_conf_preds = [{"bbox": [1, 1, 9, 9], "confidence": None, "entity_type": "valve", "point": [5, 5]}]
    result = average_precision(no_conf_preds, gt_boxes)
    check("AP is None when no predictions have confidence (Molmo2 case)", result is None)


def test_typing_accuracy_gt_crop():
    true_types = ["valve"] * 10 + ["rare_class"] * 5
    perfect_preds = list(true_types)
    result = typing_accuracy_gt_crop(perfect_preds, true_types)
    check(f"perfect typing -> accuracy=1.0 (got {result['accuracy']})", result["accuracy"] == 1.0)

    empty_result = typing_accuracy_gt_crop([], [])
    check(f"empty typing -> accuracy=0.0 (got {empty_result['accuracy']})", empty_result["accuracy"] == 0.0)

    half_preds = ["valve"] * 5 + ["wrong"] * 5 + ["rare_class"] * 5
    half_result = typing_accuracy_gt_crop(half_preds, true_types)
    check(f"half-correct typing -> sane middle (got {half_result['accuracy']})", 0.0 < half_result["accuracy"] < 1.0)
    check("rare class (5 < 20 instances) tracked in rare_class_recall", "rare_class" in half_result["rare_class_recall"])
    check(f"rare_class recall == 1.0 (got all 5 correct)", half_result["rare_class_recall"]["rare_class"] == 1.0)


def test_stitch_boundary_symbol_appears_once():
    # a symbol straddling the seam between two adjacent tiles, detected in both
    tile_a_preds = [{"bbox": [1000, 100, 1020, 120], "confidence": 0.9, "entity_type": "valve"}]
    tile_b_preds = [{"bbox": [5, 100, 25, 120], "confidence": 0.85, "entity_type": "valve"}]
    # tile_b's origin is at x=1000 (adjacent tile with 205px overlap per checklist 2.3),
    # so tile_b's local [5,100,25,120] maps to sheet [1005,100,1025,120] — same symbol
    tile_origins = [(0, 0), (1000, 0)]

    deduped, total_raw = stitch_and_dedup([tile_a_preds, tile_b_preds], tile_origins)
    check(f"stitched count <= sum of per-tile counts (2 raw -> {len(deduped)} deduped)", len(deduped) <= total_raw)
    check(f"boundary symbol appears once after dedup (got {len(deduped)})", len(deduped) == 1)
    check("dedup kept the higher-confidence duplicate", deduped[0]["confidence"] == 0.9)


def test_stitch_distinct_symbols_not_merged():
    # two genuinely different valves, far apart — must NOT be merged
    tile_preds = [
        {"bbox": [0, 0, 20, 20], "confidence": 0.9, "entity_type": "valve"},
        {"bbox": [500, 500, 520, 520], "confidence": 0.9, "entity_type": "valve"},
    ]
    deduped, _ = stitch_and_dedup([tile_preds], [(0, 0)])
    check(f"distinct far-apart symbols not merged (got {len(deduped)}, expected 2)", len(deduped) == 2)


def test_match_predictions_no_double_counting():
    # a single prediction should not be able to match two overlapping GT boxes
    gt_boxes = [[0, 0, 100, 100], [10, 10, 90, 90]]  # nested/overlapping
    preds = [{"bbox": [45, 45, 55, 55], "confidence": 0.9, "entity_type": "valve"}]
    n_matched, n_pred, n_gt = match_predictions_to_gt(preds, gt_boxes)
    check(f"one prediction matches at most one GT box (got {n_matched})", n_matched == 1)


if __name__ == "__main__":
    test_point_in_box()
    test_precision_recall_f1_confirm_bar()
    test_average_precision_none_when_no_confidence()
    test_typing_accuracy_gt_crop()
    test_stitch_boundary_symbol_appears_once()
    test_stitch_distinct_symbols_not_merged()
    test_match_predictions_no_double_counting()
    print("\nAll Phase 3 confirm-bar tests passed.")
