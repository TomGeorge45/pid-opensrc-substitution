from extraction_local.molmo_candidates import molmo_candidates


def test_point_with_nearby_word_emits_candidate():
    points_by_class = {"valve": [(100.0, 100.0)]}
    ocr_words = [("PSV-101", 90.0, 90.0, 130.0, 110.0)]
    cands, extra_boxes = molmo_candidates(points_by_class, ocr_words, radius=120)
    assert len(cands) == 1
    c = cands[0]
    assert c["raw"] == "PSV-101"
    assert c["text"] == "PSV-101"
    assert c["source"] == "molmo_point"
    assert c["shape"] == "circle"
    assert c["signals"] == ["molmo_point"]
    assert c["box"] == (90.0, 90.0, 130.0, 110.0)
    assert len(extra_boxes) == 1
    x0, y0, x1, y1 = extra_boxes[0]
    assert (x1 - x0) == 240 and (y1 - y0) == 240


def test_point_with_no_word_in_radius_emits_no_candidate_but_extends_boxes():
    points_by_class = {"valve": [(1000.0, 1000.0)]}
    ocr_words = [("FAR-AWAY", 0.0, 0.0, 10.0, 10.0)]
    cands, extra_boxes = molmo_candidates(points_by_class, ocr_words, radius=120)
    assert cands == []
    assert len(extra_boxes) == 1
    box = extra_boxes[0]
    assert box == (1000.0 - 120, 1000.0 - 120, 1000.0 + 120, 1000.0 + 120)


def test_dedupe_overlapping_tile_points():
    # Two near-identical points (e.g. from overlapping tile reads), radius/2 apart threshold.
    points_by_class = {"valve": [(100.0, 100.0), (105.0, 102.0), (500.0, 500.0)]}
    ocr_words = []
    cands, extra_boxes = molmo_candidates(points_by_class, ocr_words, radius=120)
    # (100,100) and (105,102) are within radius/2=60 of each other -> deduped to 1;
    # (500,500) is far -> kept. Total kept points = 2.
    assert len(extra_boxes) == 2


def test_nearest_word_chosen_when_multiple_in_radius():
    points_by_class = {"instrument": [(0.0, 0.0)]}
    ocr_words = [
        ("FAR", 50.0, 50.0, 60.0, 60.0),      # farther
        ("NEAR", 5.0, 5.0, 15.0, 15.0),       # closer
    ]
    cands, _ = molmo_candidates(points_by_class, ocr_words, radius=120)
    assert len(cands) == 1
    assert cands[0]["raw"] == "NEAR"


def test_pure_function_no_io_multi_class():
    points_by_class = {
        "valve": [(10.0, 10.0)],
        "pump": [(1000.0, 10.0)],
    }
    ocr_words = [
        ("V-1", 8.0, 8.0, 20.0, 18.0),
        ("P-1", 998.0, 8.0, 1010.0, 18.0),
    ]
    cands, extra_boxes = molmo_candidates(points_by_class, ocr_words, radius=120)
    assert len(cands) == 2
    assert len(extra_boxes) == 2
    texts = {c["raw"] for c in cands}
    assert texts == {"V-1", "P-1"}
