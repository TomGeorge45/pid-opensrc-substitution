"""Part B rerun (2026-07-24 Tier-1 benchmark) — same real, unmodified
ocr_reasoning_extract + apply_hierarchy pipeline as the original 2026-07-23 run, saved to
disk THIS time (the original run's JSON was never persisted -- ad hoc scratch). Re-running
because item 1 (claim partitioning) and item 4 (line-label filtering) need real extraction
data with tags/relationships to test the agreement_diff.py changes against; the old run's
raw output no longer exists anywhere. Real GPT-5.5-low API cost, ~$1.89 total across the
3 sheets last time -- Tom explicitly approved spending this again.
"""
import asyncio
import json
import os
import sys
import time

AGENT_DIR = "/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-extraction-agent"
RELBENCH_DIR = "/Users/tomgeorge/pid-ml/src/relation_bench"
for p in (AGENT_DIR, RELBENCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Set OUR key explicitly before pnid_pipeline.run._load_env's setdefault walk-up can find
# a DIFFERENT .env (rive-ai-platform/.env exists and would otherwise silently win).
for line in open("/Users/tomgeorge/pid-ml/.env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

import fitz  # noqa: E402

from pnid_pipeline import rasterize as RZ  # noqa: E402
from pnid_pipeline.triage import triage_page  # noqa: E402
from pnid_pipeline.ocr_reasoning import ocr_reasoning_extract  # noqa: E402
from pnid_pipeline.hierarchy import apply_hierarchy  # noqa: E402
from pnid_pipeline.models import DrawingMeta  # noqa: E402
from pnid_pipeline.assemble import build_result  # noqa: E402
from pnid_pipeline.run import load_config, _load_env  # noqa: E402
from arms.openai_call_llm import build_openai_call_llm  # noqa: E402
from pnid_pipeline.llm_proxy import snapshot, delta, usage_cost  # noqa: E402

SCRATCH = "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/10ffbddb-0f7d-4f19-a544-f1152513500c/scratchpad"
SHEET_PDF = sys.argv[1]
STEM = os.path.splitext(os.path.basename(SHEET_PDF))[0]
OUT_DIR = os.path.join(SCRATCH, "preds_gpt55_partB_2026-07-24")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_vector_words(pg, zoom):
    rmat = pg.rotation_matrix
    words = []
    for w in pg.get_text("words"):
        text = (w[4] or "").strip()
        if not text:
            continue
        r = (fitz.Rect(w[0], w[1], w[2], w[3]) * rmat) * zoom
        words.append((text, min(r.x0, r.x1), min(r.y0, r.y1), max(r.x0, r.x1), max(r.y0, r.y1)))
    return words


async def main():
    _load_env()
    cfg = load_config()
    call_llm = build_openai_call_llm(api_key=os.environ["OPENAI_API_KEY"])
    vr = "gpt-5.5-low"

    doc = fitz.open(SHEET_PDF)
    pg = doc[0]
    tri = triage_page(pg, 0, cfg)
    zoom = RZ.work_zoom(tri.width_pt, cfg)
    img, W, H = RZ.render_page(pg, zoom)
    words = extract_vector_words(pg, zoom)
    print(f"vector text words extracted: {len(words)}")

    before = snapshot(call_llm)
    t0 = time.monotonic()

    tags, ameta = await ocr_reasoning_extract(img, W, H, "", call_llm, vr, cfg, words=words)
    meta = DrawingMeta(source_file=os.path.basename(SHEET_PDF), page=1,
                       sheet_size=tri.sheet_size, detected_standard=(ameta.get("standard") or None),
                       page_rotation_applied_deg=tri.rotation, render_dpi=int(72 * zoom),
                       canvas_px=(W, H), pdf_points=(tri.width_pt, tri.height_pt), route="ocr_reasoning_vector_words")
    qa_extra = {"snap_rate": 0.0, "unclaimed_shapes": 0, "assertion_failures": [],
                "passes_run": ameta.get("passes", 1), "duration_seconds": 0.0, "cost_usd": 0.0}
    result = build_result(meta, tags, [], [], W, H, qa_extra)

    if cfg.get("hierarchy_pass", {}).get("enable", True):
        hmodel = "gpt-5.5-low"
        result = await apply_hierarchy(result, img, W, H, call_llm, hmodel, cfg)

    elapsed = time.monotonic() - t0
    d = delta(before, snapshot(call_llm))
    cost = usage_cost(d)
    n_calls = sum(u.get("calls", 0) for u in d.values())

    dumped = result.model_dump(by_alias=True)
    dumped["_pdf_path"] = SHEET_PDF
    out_path = os.path.join(OUT_DIR, f"{STEM}_gpt55low.json")
    with open(out_path, "w") as f:
        json.dump(dumped, f, indent=1)

    print(f"sheet: {STEM}")
    print(f"tags extracted: {len(result.tags)}")
    print(f"relationships: {len(result.relationships)}")
    print(f"elapsed: {elapsed:.1f}s, llm_calls: {n_calls}, cost_usd: ${cost}")
    print(f"errors: {result.errors}")
    print(f"saved -> {out_path}")


asyncio.run(main())
