"""Phase 1 harness: run the REAL, UNMODIFIED `pnid_pipeline.extract.extract_page`
end-to-end with local-model substitutes injected at its own extension points
(monkeypatched `tiled_ocr_words` + `vision_key` preflight, an injected `call_llm`),
write the preds JSON in the SAME shape `scripts/eval/_alt_bench.py` writes
(`res.model_dump(by_alias=True)` -> `preds/<stem>_p1.json`), then score it with
THEIR real `scripts/eval/score.py` functions.

Nothing here reimplements pipeline logic — `extract_page` runs unmodified.
`force_route=None` is REQUIRED (matches how the recorded GPT-5.5 baselines were
produced — see Extraction_Agent_Local_Plan.md §11 build-log item 2: route is
whatever `triage_page` naturally decides, never forced).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

AGENT_DIR = "/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-extraction-agent"
SCRATCH = (
    "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6"
    "/scratchpad"
)
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (AGENT_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import fitz  # noqa: E402  pymupdf

from extraction_local.molmo_candidates import (  # noqa: E402
    Word as _Word,
    _dedup_points,
    molmo_candidates,
)
from extraction_local.molmo_synthetic_tokens import molmo_synthetic_tokens  # noqa: E402
from extraction_local.paddle_ocr import paddle_ocr_words  # noqa: E402
from extraction_local.qwen_call_llm import build_qwen_call_llm  # noqa: E402

import pnid_pipeline.vision as vision  # noqa: E402
import pnid_pipeline.ocr_reasoning as ocr_reasoning  # noqa: E402
import pnid_pipeline.extract as extract_mod  # noqa: E402
from pnid_pipeline.extract import extract_page  # noqa: E402
from pnid_pipeline.run import load_config, _load_env  # noqa: E402
from scripts.eval.score import load_reviewed_truth, review_keep, review_recall  # noqa: E402

# Captured once, the FIRST time this module patches `ocr_reasoning.ocr_reasoning_extract`
# (see `_install_ocr_reasoning_molmo_wrapper` below) — lets the wrapper call through to
# the real implementation without re-wrapping itself on a second run in the same process.
_ORIG_OCR_REASONING_EXTRACT = ocr_reasoning.ocr_reasoning_extract

# Same capture pattern for the CV-hybrid path's Molmo slot: `extract.py` imported
# `snap_candidates` into its OWN namespace at module load (`from .reconcile import
# assertions, snap_candidates`, extract.py line 25) and every call to it inside
# `extract.py` (`_path_b_candidates` lines ~100/~110, `extract_page` line ~246) is a
# module-global name lookup at CALL time — so patching
# `pnid_pipeline.extract.snap_candidates` takes effect without touching agent source.
_ORIG_SNAP_CANDIDATES = extract_mod.snap_candidates

# Most recent OCR word list produced by `_paddle_tiled_ocr_words_shim` during the
# current `extract_page` run. Inside `_path_b_candidates`, OCR runs (extract.py line
# ~78, `words = await tiled_ocr_words(img, key)`) BEFORE the first `snap_candidates`
# call (line ~100) — verified by reading the real code — so by the time the CV-Molmo
# snap wrapper below fires, this holder is always populated for that run. Reset to
# None at the start of every `run_one_sheet`.
_LAST_OCR_WORDS: Dict[str, Optional[List[_Word]]] = {"words": None}


def render_pdf_page(pdf_path: str, out_png: str, dpi: int = 150) -> Tuple[int, int]:
    """Same 150dpi convention as `e2e_harness/score_revR_real_sheets.py`'s
    `render_pdf_page` — used here only for the PaddleOCR word-count sanity check;
    `extract_page` renders the page ITSELF internally (via `rasterize.render_page`
    at its own calibrated work-dpi), so this is not fed back into extract_page."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    pix.save(out_png)
    w, h = pix.width, pix.height
    doc.close()
    return w, h


def _dummy_vision_key() -> str:
    """We are not using Google Cloud Vision at all — `extract_page`'s preflight
    calls `vision_key()` and would otherwise short-circuit OCR with an empty key.
    Monkeypatched to return a non-empty dummy so the preflight passes; the actual
    OCR call is separately monkeypatched to `paddle_ocr_words` below, so this
    string is never sent anywhere."""
    return "local-paddle-no-google-vision"


async def _paddle_tiled_ocr_words_shim(img, key: str, *, tile: int = 1400, overlap: int = 220,
                                        concurrency: int = 4) -> List[_Word]:
    """Async shim wrapping the synchronous `paddle_ocr_words` so it matches
    `tiled_ocr_words`'s async signature (extract.py always `await`s it). `key` and
    `concurrency` are accepted for signature compatibility but unused — PaddleOCR
    runs locally, no API key or HTTP concurrency involved.

    Side effect: stashes its return value in `_LAST_OCR_WORDS` so the CV-path
    Molmo snap wrapper (`_install_cv_molmo_snap_wrapper`) can pair Molmo points
    against the SAME word list `_path_b_candidates` is working with — the words
    are produced inside that function and never exposed otherwise."""
    words = paddle_ocr_words(img, tile=tile, overlap=overlap)
    _LAST_OCR_WORDS["words"] = words
    return words


def install_local_monkeypatches() -> None:
    """Monkeypatch ONLY the extension points the pipeline already exposes for this
    purpose — do NOT edit agent source. `extract.py` imported `tiled_ocr_words` and
    `vision_key` directly into its own namespace (`from .vision import
    tiled_ocr_words, vision_key`), so BOTH the `pnid_pipeline.extract` module
    attributes and the `pnid_pipeline.vision` module attributes must be patched for
    the swap to take effect (Python binds names at import time).

    REAL DISCOVERY (2026-07-17, not anticipated by the plan's Phase-0 build log):
    `config.yaml`'s `pipeline.mode` default is **`ocr_reasoning`**, not `cv` — git
    history shows it has been `ocr_reasoning` since the file was added, so the
    recorded GPT-5.5/Sonnet/Gemini baselines in `history/model_comparison_*.json`
    (produced by `_alt_bench.py`, which never sets `PNID_MODE`) almost certainly
    ran under `ocr_reasoning`, NOT the `cv`+grounded-VLM path the plan's whole
    architecture (§3, §3.1 Molmo slot, `read_shapes`/`read_regions`) was designed
    against. `ocr_reasoning.py` ALSO imports `tiled_ocr_words` into its own module
    namespace (`from .vision import tiled_ocr_words`), so it needs patching too if
    that mode is ever exercised — done here for completeness, but Phase 1 forces
    `PNID_MODE=cv` (see `run_one_sheet`) to actually exercise the `grounded_read.py`
    code path this harness's Qwen/Molmo components were built for. This is a real
    open question for Tom before Phase 2: build ocr_reasoning support too (its
    `OcrResult`/`OcrTag` schema is different from `_OneTag`/`_ManyTags`), or accept
    that comparing an intentionally-forced `cv` mode against `ocr_reasoning`-mode
    baselines is no longer a clean like-for-like swap."""
    import pnid_pipeline.extract as extract_mod

    vision.tiled_ocr_words = _paddle_tiled_ocr_words_shim
    vision.vision_key = _dummy_vision_key
    extract_mod.tiled_ocr_words = _paddle_tiled_ocr_words_shim
    extract_mod.vision_key = _dummy_vision_key
    try:
        import pnid_pipeline.ocr_reasoning as ocr_reasoning_mod
        ocr_reasoning_mod.tiled_ocr_words = _paddle_tiled_ocr_words_shim
    except Exception:
        pass


def _fake_generate_fn(prompt: str, images: List[str], max_new_tokens: Optional[int] = None) -> str:
    """Phase 1 FAKE generate_fn (acceptance-test stand-in for real Qwen3-VL
    generation, injected later in Phase 2 on GPU). Detects which schema is being
    requested by prompt content (the schema's JSON Schema is appended to the
    prompt by `qwen_call_llm._schema_instruction`) and returns a plausible canned
    answer for each. Used for the `L-cv` (`_OneTag`/`_ManyTags`) path.

    `max_new_tokens` (Phase 1c fix 3(b)): accepted and ignored -- this fake has no
    real generation loop to cap, but the parameter must exist so this fake keeps
    working unchanged if `build_qwen_call_llm` is ever given a `max_new_tokens_cap`
    in this harness (not done yet; Phase 2's real Qwen wrapper is the first
    caller that will actually honor it)."""
    if '"tags"' in prompt:
        return '{"tags": ["TEST-102", "TEST-103"]}'
    if '"tag"' in prompt:
        return '{"tag": "TEST-101"}'
    return '{"tag": ""}'


def _fake_generate_fn_ocr(prompt: str, images: List[str], max_new_tokens: Optional[int] = None) -> str:
    """Phase 1b FAKE generate_fn for the `L-ocr` (`ocr_reasoning`) path. That
    path's schema is `OcrResult` (`standard`/`prefix`/`tags: List[OcrTag]`),
    distinguishable from `_ManyTags`'s flat `tags: List[str]` by the presence of
    `word_ids` in the appended JSON Schema. Cites OCR token id 0, which always
    exists whenever any OCR words were found (real PaddleOCR output on the test
    sheet has 133+ words) — a real-looking, schema-valid canned answer, not a
    plumbing shortcut.

    `max_new_tokens`: see `_fake_generate_fn`'s docstring -- accepted and ignored,
    same reasoning."""
    if '"word_ids"' in prompt:
        return (
            '{"standard": "ISA-5.1", "prefix": "", "tags": '
            '[{"text": "TEST-201", "type": "instrument", "word_ids": [0], "bbox": []}]}'
        )
    return '{"standard": "", "prefix": "", "tags": []}'


def _install_ocr_reasoning_molmo_wrapper(
    molmo_points_by_class: Optional[Dict[str, List[Tuple[float, float]]]],
) -> None:
    """Wire the Molmo2 synthetic-token slot into the `ocr_reasoning` path (plan
    §3.A). `ocr_reasoning_extract`'s own `words` override is exactly the injection
    point the plan calls for, but `extract_page` never exposes it publicly (it
    only ever sets `words` itself, under the unrelated `PNID_HYBRID_TOKENS`
    vector-text-layer feature) — so we monkeypatch `ocr_reasoning.ocr_reasoning_extract`
    itself with a thin wrapper. `extract.py` calls it as `OR.ocr_reasoning_extract(...)`
    via `from . import ocr_reasoning as OR` (a MODULE import, not a direct function
    import), so attribute lookup happens at call time and this patch takes effect
    exactly like the `tiled_ocr_words` patches above.

    When no Molmo points are supplied this installs the untouched original
    function (idempotent, safe to call every run) so `words` stays `None` and the
    real `_ocr_words` (-> patched, PaddleOCR-backed `tiled_ocr_words`) runs as-is.
    """
    if not molmo_points_by_class:
        ocr_reasoning.ocr_reasoning_extract = _ORIG_OCR_REASONING_EXTRACT
        return

    async def _wrapped(img, W, H, key, call_llm, model, cfg, words=None):
        if words is None:
            # Real OCR words first (honors _ocr_words' small-raster auto-upscale,
            # and runs through the already-patched, PaddleOCR-backed
            # tiled_ocr_words) -- these anchor the near/pair-radius pairing.
            real_words: List[_Word] = list(await ocr_reasoning._ocr_words(img, key))

            # Phase 1c fix 2: dedup raw Molmo points BEFORE synthetic-token
            # generation, reusing molmo_candidates.py's already-tested
            # `_dedup_points` rather than reimplementing it. Same convention as
            # that module: min_dist = pair_radius / 2.0, where pair_radius is
            # whatever radius `molmo_synthetic_tokens` pairs against below
            # (its default, 120px, since we don't override it here).
            pair_radius = 120.0
            deduped = _dedup_points(molmo_points_by_class, min_dist=pair_radius / 2.0)
            deduped_points_by_class = {
                cls: [(x, y) for (c, x, y) in deduped if c == cls]
                for cls in dict.fromkeys(c for (c, _x, _y) in deduped)
            }

            synth_rows = molmo_synthetic_tokens(
                deduped_points_by_class, real_words, W, H, start_id=len(real_words),
            )
            # `ocr_reasoning_extract` builds its own `idx_words` internally from a
            # `words` list of (text, x0, y0, x1, y1) tuples -- convert the
            # normalized-coordinate synthetic rows back into that raw "fake Word"
            # shape (a small synthetic box centered on the Molmo point) so they go
            # through the SAME internal id-assignment/normalization loop as real
            # OCR words, rather than trying to splice pre-normalized rows into
            # `idx_words` after the fact.
            half = 4.0  # px -- small synthetic box, just needs a non-zero center
            synth_words: List[_Word] = [
                (text, cx_n * W - half, cy_n * H - half, cx_n * W + half, cy_n * H + half)
                for (_id, text, cx_n, cy_n) in synth_rows
            ]
            words = real_words + synth_words
        return await _ORIG_OCR_REASONING_EXTRACT(
            img, W, H, key, call_llm, model, cfg, words=words,
        )

    ocr_reasoning.ocr_reasoning_extract = _wrapped


def _install_cv_molmo_snap_wrapper(
    molmo_points_by_class: Optional[Dict[str, List[Tuple[float, float]]]],
) -> None:
    """Wire the Molmo2 slot into the CV-hybrid path (plan §3.B) with ZERO agent-source
    edits, by monkeypatching `pnid_pipeline.extract.snap_candidates` (a module-global
    name lookup at call time — see `_ORIG_SNAP_CANDIDATES`'s comment above).

    Mechanism, all verified by reading `extract.py` directly:
    - `snap_candidates` is called up to 3 times per run: `_path_b_candidates` line ~100
      (first merge of shape+region+ocr candidates), line ~110 (adjudication re-snap),
      and `extract_page` line ~246 (A+B combined re-snap). The FIRST call of any run
      that has symbols is always line ~100 — routes "B" and "A+B" both enter
      `_path_b_candidates` (extract.py line ~237-239) BEFORE line ~246 runs, and line
      ~246 is skipped entirely (`if symbols and R`) when Path B never ran. So a simple
      one-shot closure flag injects exactly once, at the real merge point.
    - On that first call the wrapper appends `molmo_candidates(...)`'s candidate dicts
      to the merge input AND extends the `symbols` list IN PLACE (never rebinding it):
      `symbols` is the list object owned by `_path_b_candidates`, so the same object
      later flows into `assertions(symbols, cands, R)` (line ~104) and the adjudication
      loop — the Molmo boxes count as real symbol geometry there, exactly the plan §3.B
      design ("append Molmo2-found symbol boxes to `symbols` so (a) snap can
      corroborate, (b) `assertions()` counts them for the adjudication pass").
    - OCR words come from `_LAST_OCR_WORDS`, stashed by `_paddle_tiled_ocr_words_shim`
      — OCR runs at line ~78, before the first snap call, so they are always fresh.

    Coordinate space (verified): `_path_b_candidates` renders via
    `zoom = RZ.work_zoom(tri.width_pt, cfg); img, W, H = RZ.render_page(pg, zoom)`
    (extract.py lines ~74-75) — the EXACT same sequence `molmo_render.molmo_render_page`
    replicates (see that module's docstring), so Phase A's cached Molmo points are
    already in this path's pixel space. Same contract as `molmo_points.py`.

    When no Molmo points are supplied this restores the untouched original
    (idempotent, safe to call every run) — same pattern as
    `_install_ocr_reasoning_molmo_wrapper` above."""
    if not molmo_points_by_class:
        extract_mod.snap_candidates = _ORIG_SNAP_CANDIDATES
        return

    injected = {"done": False}   # one-shot flag; wrapper reinstalled fresh per run

    def _wrapped(cands, symbols, R):
        if not injected["done"]:
            injected["done"] = True
            ocr_words = _LAST_OCR_WORDS["words"] or []
            m_cands, extra_symbol_boxes = molmo_candidates(
                molmo_points_by_class, ocr_words, radius=120,
            )
            # IN PLACE — same list object must flow on to assertions()/adjudication.
            symbols.extend([list(b) for b in extra_symbol_boxes])
            cands = list(cands) + m_cands
        return _ORIG_SNAP_CANDIDATES(cands, symbols, R)

    extract_mod.snap_candidates = _wrapped


def run_one_sheet(
    pdf_path: str,
    stem: str,
    *,
    pipeline_mode: str = "cv",
    generate_fn: Optional[Callable[[str, List[str]], str]] = None,
    molmo_points_by_class: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    preds_dir: Optional[str] = None,
) -> Dict:
    """Full pipeline run on one sheet, through EITHER real pipeline code path:

    - `pipeline_mode="cv"` (`L-cv`, Phase 1's original build): the CV-hybrid
      `read_shapes`/`read_regions` path (`grounded_read.py`), forced via
      `PNID_MODE=cv`. `molmo_points_by_class`, if given, IS wired in for this
      mode (plan §3.B): `_install_cv_molmo_snap_wrapper` monkeypatches
      `pnid_pipeline.extract.snap_candidates` with a one-shot wrapper that, on
      the FIRST snap call of the run (the shape+region+ocr merge inside
      `_path_b_candidates`), appends `molmo_candidates(...)`'s candidate dicts
      to the merge and extends the `symbols` list in place with the Molmo
      synthetic boxes — see that installer's docstring for the verified
      mechanism. Zero agent-source edits.
    - `pipeline_mode="ocr_reasoning"` (`L-ocr`, Phase 1b, the TRUE apples-to-apples
      config vs. the recorded GPT-5.5 baselines): the real DEFAULT pipeline mode
      (`config.yaml`: `mode: ocr_reasoning`) — `PNID_MODE` is left UNSET here (no
      override), so `extract_page` takes this branch purely because it's the
      real default, not because we forced it. `molmo_points_by_class`, if given,
      IS wired in for this mode: `_install_ocr_reasoning_molmo_wrapper` injects
      Molmo2 points as synthetic OCR tokens via `ocr_reasoning_extract`'s own
      `words` override (see that function's docstring for the exact mechanism).
    """
    if pipeline_mode not in ("cv", "ocr_reasoning"):
        raise ValueError(f"pipeline_mode must be 'cv' or 'ocr_reasoning', got {pipeline_mode!r}")

    install_local_monkeypatches()
    _install_ocr_reasoning_molmo_wrapper(
        molmo_points_by_class if pipeline_mode == "ocr_reasoning" else None
    )
    _install_cv_molmo_snap_wrapper(
        molmo_points_by_class if pipeline_mode == "cv" else None
    )
    _LAST_OCR_WORDS["words"] = None   # no stale carryover between sheets/runs

    _load_env()
    cfg = load_config()

    # Phase 1c fix 3(a): lower the ocr_reasoning dense-chunk size (real config key,
    # confirmed by reading ocr_reasoning.py directly: `int(cfg.get("ocr_reasoning", {})
    # .get("dense_chunk_tokens", 1500))`, ~line 329) so each reasoning call covers
    # fewer OCR tokens -- shorter completions, lower truncation risk, faster per-call
    # latency on local serial GPU generation. A plain dict mutation on the loaded
    # config, NOT an edit to config.yaml itself.
    cfg.setdefault("ocr_reasoning", {})["dense_chunk_tokens"] = 800

    if pipeline_mode == "cv":
        # See install_local_monkeypatches()'s docstring: config.yaml's real default
        # pipeline.mode is "ocr_reasoning", not "cv". Phase 1's four components
        # (build_qwen_call_llm, paddle_ocr_words, molmo_candidates, this harness)
        # were built against the "cv"-mode grounded_read.py contract, so we force
        # PNID_MODE=cv here (a MODE override via env var, not a ROUTE override —
        # force_route stays None; route A/B/A+B is still whatever triage_page
        # naturally decides). Explicit assignment (not setdefault) so a prior
        # "ocr_reasoning" run in the same process can't leave this stale.
        os.environ["PNID_MODE"] = "cv"
    else:
        # L-ocr must run through the REAL default with NO env-var deviation —
        # explicitly clear any leftover override from a prior "cv" run in the
        # same process, rather than relying on it never having been set.
        os.environ.pop("PNID_MODE", None)

    fn = generate_fn or (_fake_generate_fn if pipeline_mode == "cv" else _fake_generate_fn_ocr)
    call_llm = build_qwen_call_llm(fn)

    default_dir = "preds_local" if pipeline_mode == "cv" else "preds_local_ocr"
    preds_dir = preds_dir or os.path.join(SCRATCH, default_dir)
    os.makedirs(preds_dir, exist_ok=True)

    result = asyncio.run(extract_page(pdf_path, 0, cfg, call_llm, force_route=None))

    dumped = result.model_dump(by_alias=True)
    out_path = os.path.join(preds_dir, f"{stem}_p1.json")
    with open(out_path, "w") as f:
        json.dump(dumped, f, indent=1)

    return {"result": result, "dumped": dumped, "preds_path": out_path, "call_llm": call_llm,
            "pipeline_mode": pipeline_mode, "pnid_mode_env": os.environ.get("PNID_MODE")}


def score_against_reviewed_truth(stem: str, dumped: Dict) -> Dict:
    truth_path = os.path.join(AGENT_DIR, "scripts", "eval", "review_reads", stem, "reviewed_truth.json")
    truth = load_reviewed_truth(truth_path)
    pred = {a for a in (review_keep(t.get("text")) for t in dumped.get("tags", [])) if a}
    revR, hits, missed = review_recall(pred, truth)
    return {
        "stem": stem, "revR": round(revR, 3), "hits": hits, "n_truth": len(truth),
        "n_pred_clean": len(pred), "missed": missed,
    }


def main() -> None:
    stem = "GD-T-435-DT-2042-056"
    pdf_path = os.path.join(SCRATCH, "AG_PNID", "AG_PNID", "GD-T-435-DT-2042-056-Z.pdf")

    # Word-count sanity check for the PaddleOCR adapter (plan §5 item 1) — render at
    # 150dpi (same convention as score_revR_real_sheets.py) and OCR it directly; not
    # fed into extract_page (which renders internally at its own calibrated dpi).
    png_path = os.path.join(SCRATCH, f"{stem}_render.png")
    if not os.path.isfile(png_path):
        render_pdf_page(pdf_path, png_path)
    import cv2
    img = cv2.imread(png_path)
    words = paddle_ocr_words(img)
    print(f"[sanity] PaddleOCR word count on {stem} @150dpi: {len(words)}")

    # Fake Molmo points (acceptance-test placeholder — wired into BOTH paths now:
    # cv via _install_cv_molmo_snap_wrapper, ocr_reasoning via
    # _install_ocr_reasoning_molmo_wrapper).
    fake_molmo_points = {
        "valve": [(400.0, 500.0), (405.0, 503.0)],   # 2nd is a near-dupe (tile overlap)
        "instrument_bubble": [(1200.0, 800.0)],
    }

    print(f"\n=== L-cv: extract_page (force_route=None, PNID_MODE=cv) on {stem} ===")
    out = run_one_sheet(pdf_path, stem, pipeline_mode="cv", molmo_points_by_class=fake_molmo_points)
    dumped = out["dumped"]
    print(f"route={dumped['drawing'].get('route')}  n_tags={len(dumped.get('tags', []))}")
    print(f"preds written to {out['preds_path']}")
    print(f"call_llm.usage = {out['call_llm'].usage}")

    scored = score_against_reviewed_truth(stem, dumped)
    print(f"\nrevR = {scored['revR']} ({scored['hits']}/{scored['n_truth']} truth tags found, "
          f"{scored['n_pred_clean']} distinct cleaned predicted tags)")
    if scored["missed"]:
        print(f"missed: {scored['missed']}")

    print(f"\n=== L-ocr: extract_page (force_route=None, PNID_MODE UNSET -> real default "
          f"'ocr_reasoning') on {stem} ===")
    out2 = run_one_sheet(pdf_path, stem, pipeline_mode="ocr_reasoning",
                         molmo_points_by_class=fake_molmo_points)
    dumped2 = out2["dumped"]
    print(f"PNID_MODE env after run: {out2['pnid_mode_env']!r} (must be None -- no override)")
    print(f"route={dumped2['drawing'].get('route')}  n_tags={len(dumped2.get('tags', []))}")
    print(f"preds written to {out2['preds_path']}")
    print(f"call_llm.usage = {out2['call_llm'].usage}")

    scored2 = score_against_reviewed_truth(stem, dumped2)
    print(f"\nrevR = {scored2['revR']} ({scored2['hits']}/{scored2['n_truth']} truth tags found, "
          f"{scored2['n_pred_clean']} distinct cleaned predicted tags)")
    if scored2["missed"]:
        print(f"missed: {scored2['missed']}")


if __name__ == "__main__":
    main()
