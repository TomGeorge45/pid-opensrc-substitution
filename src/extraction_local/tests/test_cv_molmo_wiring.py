"""CV-path Molmo wiring (`_install_cv_molmo_snap_wrapper`, plan §3.B): the
`pnid_pipeline.extract.snap_candidates` monkeypatch must (a) inject Molmo
candidates on the FIRST snap call of a run only, (b) extend the `symbols` list
IN PLACE (same object identity — `_path_b_candidates` later feeds that same
object into `assertions()`), (c) leave subsequent calls untouched, and
(d) restore the captured original when installed with None.

Same style as `test_molmo_dedup_wiring.py`: monkeypatch the `_ORIG_*` capture
on the harness module with a fake that records calls, install the wrapper,
drive `extract_mod.snap_candidates` directly."""
import extraction_local.run_extraction_local as rel
import pnid_pipeline.extract as extract_mod

# One fake OCR word within 120px of the valve point (400, 500) -> a molmo_point
# candidate; the instrument_bubble point (1200, 800) has no word in radius ->
# synthetic symbol box only, no candidate.
FAKE_OCR_WORDS = [("PV-101", 380.0, 480.0, 420.0, 495.0)]
FAKE_MOLMO_POINTS = {
    "valve": [(400.0, 500.0)],
    "instrument_bubble": [(1200.0, 800.0)],
}


def _setup(monkeypatch):
    calls = []

    def _fake_orig_snap(cands, symbols, R):
        calls.append({"cands": list(cands), "symbols": symbols, "R": R})
        return cands

    # Register extract_mod.snap_candidates for teardown restoration BEFORE the
    # installer overwrites it, then swap the harness's captured original for the
    # recording fake (global lookup at call time, same trick as the dedup test).
    monkeypatch.setattr(extract_mod, "snap_candidates", extract_mod.snap_candidates)
    monkeypatch.setattr(rel, "_ORIG_SNAP_CANDIDATES", _fake_orig_snap)
    monkeypatch.setitem(rel._LAST_OCR_WORDS, "words", list(FAKE_OCR_WORDS))
    return calls, _fake_orig_snap


def test_first_call_injects_molmo_candidates_and_extends_symbols_in_place(monkeypatch):
    calls, _ = _setup(monkeypatch)
    rel._install_cv_molmo_snap_wrapper(FAKE_MOLMO_POINTS)

    base_cands = [{"text": "X", "box": (0, 0, 1, 1), "signals": ["ocr"]}]
    symbols = [[10.0, 10.0, 20.0, 20.0]]

    out = extract_mod.snap_candidates(base_cands, symbols, 40)

    # (a) exactly one molmo_point candidate injected (valve point paired with the
    # word; bubble point had no word in radius -> box only, no candidate).
    seen = calls[0]["cands"]
    molmo = [c for c in seen if c.get("source") == "molmo_point"]
    assert len(seen) == 2 and len(molmo) == 1, seen
    assert molmo[0]["raw"] == "PV-101"
    assert "molmo_point" in molmo[0]["signals"]
    assert molmo[0] in out  # wrapper's return is the (fake) original's return

    # (b) symbols extended IN PLACE: same object identity, 2 synthetic boxes added
    # (one per deduped point — misses contribute a box even without a candidate).
    assert calls[0]["symbols"] is symbols
    assert len(symbols) == 3, symbols

    # original cands list was not mutated (wrapper builds a new list).
    assert len(base_cands) == 1


def test_second_call_gets_no_injection(monkeypatch):
    calls, _ = _setup(monkeypatch)
    rel._install_cv_molmo_snap_wrapper(FAKE_MOLMO_POINTS)

    symbols = []
    extract_mod.snap_candidates([], symbols, 40)            # first: injects
    n_symbols_after_first = len(symbols)
    extract_mod.snap_candidates([{"text": "Y", "box": (0, 0, 1, 1)}], symbols, 40)

    assert len(calls) == 2
    assert all(c.get("source") != "molmo_point" for c in calls[1]["cands"])
    assert len(symbols) == n_symbols_after_first   # no second extension


def test_installer_with_none_restores_original(monkeypatch):
    _, fake_orig = _setup(monkeypatch)
    rel._install_cv_molmo_snap_wrapper(FAKE_MOLMO_POINTS)
    assert extract_mod.snap_candidates is not fake_orig

    rel._install_cv_molmo_snap_wrapper(None)
    assert extract_mod.snap_candidates is rel._ORIG_SNAP_CANDIDATES

    # Idempotent: calling again with None is safe and stays restored.
    rel._install_cv_molmo_snap_wrapper(None)
    assert extract_mod.snap_candidates is rel._ORIG_SNAP_CANDIDATES
