"""Confirm-bar test for training-format targets: a target string built for training must be
parseable by the EXACT SAME parser used for scoring model output — otherwise training and
eval are quietly inconsistent with each other, which would be a subtle and dangerous bug to
ship. Run with:
    python3 -m src.stage4_symbol_detection.test_training_format
No GPU/Colab needed — pure string round-trip checks.
"""

from . import molmo_candidate, qwen_candidate, training_format


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label


def test_boxjson_roundtrips_through_qwen_parser():
    boxes = [
        {"bbox": [10, 20, 30, 40], "entity_type": "valve"},
        {"bbox": [100, 200, 150, 260], "entity_type": "symbol"},
    ]
    target_text = training_format.format_boxjson_target(boxes)
    detections, error = qwen_candidate.parse(target_text)

    check(f"boxjson target parses without error (error={error})", error is None)
    check(f"boxjson target round-trips exact count (got {len(detections)}, want 2)", len(detections) == 2)
    check("first box coords match input exactly",
          detections[0]["bbox"] == [10.0, 20.0, 30.0, 40.0])
    check("first box entity_type matches input",
          detections[0]["entity_type"] == "valve")
    check("ground-truth confidence is 1.0 (not a model guess)",
          all(d["confidence"] == 1.0 for d in detections))


def test_boxjson_empty_case():
    target_text = training_format.format_boxjson_target([])
    detections, error = qwen_candidate.parse(target_text)
    check(f"empty boxjson target parses to empty list (error={error})", error is None and detections == [])


def test_points_roundtrips_through_molmo_parser():
    img_w, img_h = 1000, 1000
    boxes = [
        {"bbox": [100, 100, 200, 200], "entity_type": "symbol"},  # center (150, 150) -> scaled (150, 150)
        {"bbox": [400, 600, 500, 700], "entity_type": "symbol"},  # center (450, 650) -> scaled (450, 650)
    ]
    target_text = training_format.format_points_target(boxes, img_w, img_h)
    detections, error = molmo_candidate.parse(target_text, img_w, img_h)

    check(f"points target parses without error (error={error})", error is None)
    check(f"points target round-trips exact count (got {len(detections)}, want 2)", len(detections) == 2)

    p0 = detections[0]["point"]
    check(f"first point recovers correct center (got {p0}, want ~[150,150])",
          abs(p0[0] - 150) < 1 and abs(p0[1] - 150) < 1)


def test_points_empty_case():
    target_text = training_format.format_points_target([], 1000, 1000)
    detections, error = molmo_candidate.parse(target_text, 1000, 1000)
    check(f"empty points target parses without error (error={error})", error is None)
    check(f"empty points target yields zero detections (got {len(detections) if detections else 0})",
          not detections)


def test_gupta_boxes_are_class_agnostic():
    label_lines = "0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.05 0.05\n"
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        label_path = Path(tmp) / "sheet.txt"
        label_path.write_text(label_lines)
        boxes = training_format.boxes_from_label_file(label_path, 1000, 1000, source="gupta")
        check(f"all Gupta boxes get generic entity_type (got {[b['entity_type'] for b in boxes]})",
              all(b["entity_type"] == "symbol" for b in boxes))


def test_kaggle_boxes_use_class_names_when_available():
    label_lines = "1 0.5 0.5 0.1 0.1\n"
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        label_path = Path(tmp) / "tile.txt"
        label_path.write_text(label_lines)
        boxes = training_format.boxes_from_label_file(
            label_path, 1000, 1000, source="kaggle", class_names={"1": "Gate_Valve"}
        )
        check(f"kaggle box uses real class name (got {boxes[0]['entity_type']})",
              boxes[0]["entity_type"] == "Gate_Valve")


if __name__ == "__main__":
    test_boxjson_roundtrips_through_qwen_parser()
    test_boxjson_empty_case()
    test_points_roundtrips_through_molmo_parser()
    test_points_empty_case()
    test_gupta_boxes_are_class_agnostic()
    test_kaggle_boxes_use_class_names_when_available()
    print("\nAll training-format round-trip tests passed.")
