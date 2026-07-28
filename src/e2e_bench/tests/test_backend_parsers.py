"""Fixture tests for every backend parser, including the malformed-input cases that must
fail (D10: parse failures must be counted, never silently treated as real data)."""
from e2e_bench.backends.parse_json_common import (
    parse_entity_verdict_json,
    parse_relation_verdict_json,
    parse_skid_json,
    parse_titleblock_json,
)
from e2e_bench.backends.parse_molmo import parse_molmo_points
from e2e_bench.backends.parse_paddle import parse_paddle_ocr_result


def test_molmo_real_format():
    r = parse_molmo_points('<points coords="1 500 500; 1 200 300"/>', 1024, 1024, entity_type="valve")
    assert not r.parse_failed
    assert len(r.value) == 2
    assert r.value[0].entity_type == "valve"
    assert r.value[0].confidence == 0.5  # D3 fixed default, never 0.0


def test_molmo_duplicated_leading_index_repair():
    # the documented glitch: first point has a duplicated leading index
    r = parse_molmo_points('<points coords="1 1 027 016 2 176 158"/>', 1000, 1000, entity_type="valve")
    assert not r.parse_failed
    assert len(r.value) == 2


def test_molmo_malformed_coords_fails():
    r = parse_molmo_points('<points coords="1 2"/>', 1000, 1000, entity_type="valve")
    assert r.parse_failed


def test_molmo_no_points_is_a_valid_empty_answer():
    # matches the original parser's behavior: no point-like tags at all is a legitimate
    # "found nothing" answer, not a parse failure - only "has <point but regex mismatched" fails
    r = parse_molmo_points("nothing found here", 1000, 1000, entity_type="valve")
    assert not r.parse_failed
    assert r.value == []


def test_titleblock_fenced_json():
    r = parse_titleblock_json(
        '```json\n{"drawing_number": "PIP-01-101", "revision": null, '
        '"title": "Test", "site": "X"}\n```'
    )
    assert not r.parse_failed
    assert r.value.fields["drawing_number"] == "PIP-01-101"
    assert r.value.located is True


def test_titleblock_all_null_is_not_located():
    r = parse_titleblock_json('{"drawing_number": null, "revision": null, "title": null, "site": null}')
    assert not r.parse_failed
    assert r.value.located is False


def test_titleblock_malformed_fails():
    r = parse_titleblock_json("I could not find a title block.")
    assert r.parse_failed


def test_skid_json_top_level_array_of_objects():
    # regression test for the real bug caught during this build: the loose-JSON extractor
    # used to try "{" before "[" unconditionally, truncating a top-level array down to its
    # first nested object.
    r = parse_skid_json(
        '[{"target_temp_id": "p0_e001", "forward_relation_name": "Installed Valves", '
        '"confidence": 0.9, "reasoning": "close"}]',
        "p0_e000",
    )
    assert not r.parse_failed
    assert r.value.asset_temp_id == "p0_e000"
    assert len(r.value.members) == 1
    assert r.value.members[0].target_temp_id == "p0_e001"


def test_skid_json_not_a_list_fails():
    r = parse_skid_json('{"not": "a list"}', "p0_e000")
    assert r.parse_failed


def test_entity_verdict_json_form():
    r = parse_entity_verdict_json('{"keep": false, "confidence": 0.8, "reasoning": "fake"}', "p0_e005")
    assert not r.parse_failed
    assert r.value.keep is False


def test_entity_verdict_plain_text_fallback():
    r = parse_entity_verdict_json("keep", "p0_e005")
    assert not r.parse_failed
    assert r.value.keep is True

    r2 = parse_entity_verdict_json("remove this one", "p0_e005")
    assert not r2.parse_failed
    assert r2.value.keep is False


def test_entity_verdict_ambiguous_text_fails():
    r = parse_entity_verdict_json("keep or remove, unclear", "p0_e005")
    assert r.parse_failed


def test_relation_verdict_json_form():
    r = parse_relation_verdict_json(
        '{"verdict": "rejected", "revised_confidence": 0.1, "reasoning": "no line"}', "p0_r001"
    )
    assert not r.parse_failed
    assert r.value.verdict == "rejected"


def test_relation_verdict_plain_yes_no_fallback():
    r = parse_relation_verdict_json("yes, connected", "p0_r001")
    assert not r.parse_failed
    assert r.value.verdict == "confirmed"

    r2 = parse_relation_verdict_json("no relation here", "p0_r001")
    assert not r2.parse_failed
    assert r2.value.verdict == "rejected"


def test_paddle_ocr_result_mapping():
    r = parse_paddle_ocr_result({
        "rec_texts": ["VALVE-101"], "rec_scores": [0.98], "rec_boxes": [[10, 10, 50, 30]],
    })
    assert not r.parse_failed
    assert r.value[0].text == "VALVE-101"
    assert r.value[0].bbox == [10, 10, 50, 30]


def test_paddle_ocr_empty_result():
    r = parse_paddle_ocr_result(None)
    assert not r.parse_failed
    assert r.value == []


def test_paddle_ocr_malformed_result_fails():
    r = parse_paddle_ocr_result({"rec_texts": ["a"]})  # missing rec_scores/rec_boxes
    assert r.parse_failed
