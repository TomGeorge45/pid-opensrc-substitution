"""Phase 1c fix 2: `_install_ocr_reasoning_molmo_wrapper` must dedup raw Molmo
points (reusing `molmo_candidates._dedup_points`) BEFORE calling
`molmo_synthetic_tokens`, so a near-duplicate point (e.g. from tile overlap)
doesn't produce two synthetic OCR tokens for the same real-world symbol.

Uses the SAME fake-Molmo-points acceptance-test data already in
`run_extraction_local.main()`: a deliberate 2nd near-dupe point at (405.0, 503.0)
vs (400.0, 500.0) for the "valve" class."""
import asyncio

import extraction_local.run_extraction_local as rel
import pnid_pipeline.ocr_reasoning as ocr_reasoning


def test_near_duplicate_molmo_points_collapse_to_one_synthetic_token(monkeypatch):
    captured = {}

    async def _fake_ocr_words(img, key):
        return []  # no real OCR tokens -> every kept Molmo point becomes synthetic

    async def _fake_orig_extract(img, W, H, key, call_llm, model, cfg, words=None):
        captured["words"] = words
        return [], {"standard": "", "ocr_words": len(words or [])}

    monkeypatch.setattr(ocr_reasoning, "_ocr_words", _fake_ocr_words)
    monkeypatch.setattr(rel, "_ORIG_OCR_REASONING_EXTRACT", _fake_orig_extract)

    fake_molmo_points = {
        "valve": [(400.0, 500.0), (405.0, 503.0)],   # 2nd is a near-dupe (tile overlap)
        "instrument_bubble": [(1200.0, 800.0)],
    }

    rel._install_ocr_reasoning_molmo_wrapper(fake_molmo_points)
    try:
        result, meta = asyncio.run(
            ocr_reasoning.ocr_reasoning_extract(
                img=None, W=2000, H=2000, key="dummy", call_llm=None,
                model="qwen3-vl-8b-local", cfg={}, words=None,
            )
        )
    finally:
        # Restore the untouched original so later tests/runs aren't affected.
        rel._install_ocr_reasoning_molmo_wrapper(None)

    words = captured["words"]
    # 0 real OCR words + exactly 2 synthetic ones: the deduped valve point and the
    # instrument_bubble point -- NOT 3 (which would mean the near-dupe leaked
    # through undeduped).
    assert len(words) == 2, words
