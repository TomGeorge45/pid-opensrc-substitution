"""
Scores Arm P (GPT-5.5-low, real prompt/tool-schema/tiling/PaddleOCR) against the REAL,
already-existing human-reviewed ground truth found in pnid-extraction-agent's eval harness
(rive-ai-platform/agents/pnid-extraction-agent/scripts/eval/review_reads/<stem>/reviewed_truth.json),
for the real sheets in AG_PNID.zip and RIVE_LTTS_Sample.zip.

IMPORTANT SCOPE NOTE: this ground truth is a bare list of tag TEXT strings (Claude's by-eye
reading of the drawing) - not boxes, not entity types, not relations. It only supports the
"revR" metric (prefix-aware recall against that tag list) - the SAME metric and matching
logic pnid-extraction-agent's own score.py uses, copied verbatim below so results are directly
comparable to their real recorded history (mean_revR: gpt-5.5 high=0.836, gpt-5.5 low=0.813,
across all 14 sheets, per history/model_comparison_1782539273.json).

This does NOT run pnid-intelligence-agent's stage 6/11/13/12 - revR only cares about the raw
set of tag strings GPT-5.5 proposed anywhere across all tiles (real stage 4 detection only),
so building entities/relations would add nothing this metric measures.
"""
import base64
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import fitz  # pymupdf
from PIL import Image

from e2e_harness import poc_run_arm_p_v2 as v2
from pnid_agent.sub_agents.symbol_detection.tile_words import slice_words_to_tile
from pnid_agent.stages.tile_segmentation.grid import compute_tile_grid

Image.MAX_IMAGE_PIXELS = None

# ── revR scoring, copied verbatim from pnid-extraction-agent/scripts/eval/score.py ──────────
_BARE_SIZE = re.compile(
    r'^\d+(?:[-\s]\d+/\d+)?\s*["”″]'
    r'(?:\s*[xX×]\s*\d+(?:[-\s]\d+/\d+)?\s*["”″])?$'
)
_REV_LINENO = re.compile(r"^\d{5,}[\d\-]*$")
_REV_ANNOT = re.compile(r"(SET\s*@|MAWP|DESIGN|\bGPM\b|#|\bFROM\b|\bTO\b|\bNOTE\b|DETAIL|\bROOM\b|"
                        r"\bTYP\b|\bSHEET\b|\bSIM\b|HEADER|PUMP$|DRUM$|CAISSON|BIOCIDE|NITROGEN)", re.I)


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def review_keep(text):
    t = (text or "").strip()
    a = _alnum(t)
    if len(a) < 2 or _BARE_SIZE.match(t) or _REV_ANNOT.search(t):
        return None
    if _REV_LINENO.match(re.sub(r"[^0-9\-]", "", t)) and not re.search(r"[A-Za-z]", t):
        return None
    return a


def load_reviewed_truth(path):
    d = json.load(open(path))
    tags = d.get("tags", []) if isinstance(d, dict) else d
    return set(tags)


def review_recall(pred_alnum, truth_alnum):
    missed = []
    hit = 0
    for v in truth_alnum:
        if v in pred_alnum or (len(v) >= 5 and any(p.endswith(v) or v.endswith(p)
                                                    for p in pred_alnum if len(p) >= 5)):
            hit += 1
        else:
            missed.append(v)
    return (hit / len(truth_alnum) if truth_alnum else 0.0), hit, sorted(missed)


# ── sheet -> real PDF file mapping ───────────────────────────────────────────────────────────
SCRATCH = "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6/scratchpad"
AG_DIR = f"{SCRATCH}/AG_PNID/AG_PNID"
RIVE_DIR = f"{SCRATCH}/RIVE_LTTS_Sample/RIVE"
REVIEW_DIR = ("/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/"
              "pnid-extraction-agent/scripts/eval/review_reads")

SHEETS = [
    ("GD-B-540-DP-2920-005", f"{AG_DIR}/GD-B-540-DP-2920-005-Z.pdf"),
    ("GD-B-550-DP-3322-003", f"{AG_DIR}/GD-B-550-DP-3322-003-Z2.pdf"),
    ("GD-B-615-DP-1148-006", f"{AG_DIR}/GD-B-615-DP-1148-006-Z2.pdf"),
    ("GD-H-375-DP-2590-003", f"{AG_DIR}/GD-H-375-DP-2590-003-Zpdf.pdf"),
    ("GD-T-435-DR-2031-030", f"{AG_DIR}/GD-T-435-DR-2031-030-Z2.pdf"),
    ("GD-T-435-DT-2042-056", f"{AG_DIR}/GD-T-435-DT-2042-056-Z.pdf"),
    ("PX-2365-0140006-001", f"{RIVE_DIR}/PX-2365-0140006-001.PDF"),
    ("PX-2365-0140031-001", f"{RIVE_DIR}/PX-2365-0140031-001.PDF"),
    ("PX-2365-0150022-001", f"{RIVE_DIR}/PX-2365-0150022-001.pdf"),
    ("PX-2365-0150033-008", f"{RIVE_DIR}/PX-2365-0150033-008.pdf"),
    ("PX-2365-9850077-001", f"{RIVE_DIR}/PX-2365-9850077-001.pdf"),
    ("PX-2368-0180004-001", f"{RIVE_DIR}/PX-2368-0180004-001.pdf"),
    ("PX-2368-0180021-002", f"{RIVE_DIR}/PX-2368-0180021-002.pdf"),
]


def render_pdf_page(pdf_path, out_png, dpi=150):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    pix.save(out_png)
    doc.close()


def detect_all_tags(png_path):
    """Real stage-4-only pass: real tiling + real prompt/tool schema + PaddleOCR + GPT-5.5.
    Returns the raw set of every non-null `value` string proposed across all tiles - no
    NMS/entity-assembly, since revR only needs the union of tag text ever proposed."""
    schemas = v2.build_benchmark_schemas()
    ontology_rendering = v2.render_pid_ontology_for_prompt(schemas)
    system_prompt = v2.build_system_prompt(ontology_rendering)
    system_prompt = system_prompt.replace('entity_type "connector"', 'entity_type "inlet_outlet"')
    anthropic_tool_schema = v2.build_detection_tool_schema(schemas)
    openai_tool = v2.anthropic_tool_to_openai(anthropic_tool_schema)

    print("  running PaddleOCR...")
    page_words = v2.run_paddle_ocr(png_path)
    print(f"  {len(page_words)} OCR words")

    img = Image.open(png_path).convert("RGB")
    W, H = img.size
    tiles = compute_tile_grid(drawing_bbox=[0, 0, W, H], page_size=(W, H))
    print(f"  {len(tiles)} tiles ({W}x{H})")

    raw_values = set()
    for t in tiles:
        crop = img.crop((t.x0, t.y0, t.x1, t.y1))
        words_in_tile = slice_words_to_tile(page_words, tile_bbox=[t.x0, t.y0, t.x1, t.y1], margin_px=24)
        tile_id = f"p0_t{t.idx:03d}"
        payload = v2.gpt_detect_tile(system_prompt, crop, tile_id, (t.x0, t.y0), words_in_tile,
                                      openai_tool, page_index=0)
        n_before = len(raw_values)
        for spec in (payload.get("symbols") or []):
            if isinstance(spec, dict):
                v = spec.get("value") or spec.get("name")
                if v:
                    raw_values.add(str(v))
        print(f"    tile {t.idx}: {len(words_in_tile)} ocr words, "
              f"+{len(raw_values) - n_before} new tag values (running total {len(raw_values)})")
    return raw_values


def main():
    results = []
    for stem, pdf_path in SHEETS:
        truth_path = f"{REVIEW_DIR}/{stem}/reviewed_truth.json"
        if not os.path.isfile(pdf_path) or not os.path.isfile(truth_path):
            print(f"SKIP {stem}: missing pdf or ground truth")
            continue
        print(f"\n=== {stem} ===")
        png_path = f"{SCRATCH}/{stem}_render.png"
        if not os.path.isfile(png_path):
            render_pdf_page(pdf_path, png_path)

        raw_values = detect_all_tags(png_path)
        pred_clean = {a for a in (review_keep(v) for v in raw_values) if a}
        truth = load_reviewed_truth(truth_path)
        rr, hits, missed = review_recall(pred_clean, truth)
        print(f"  revR = {rr:.3f}  ({hits}/{len(truth)} truth tags found, "
              f"{len(pred_clean)} distinct cleaned predicted tags)")
        results.append({"stem": stem, "revR": rr, "hits": hits, "n_truth": len(truth),
                         "n_pred_clean": len(pred_clean)})
        with open("/tmp/revR_results.json", "w") as f:
            json.dump(results, f, indent=2)

    mean_revR = sum(r["revR"] for r in results) / len(results) if results else 0.0
    print(f"\n{'='*60}\nMEAN revR across {len(results)} sheets: {mean_revR:.3f}")
    print("Reference (pnid-extraction-agent history, 14 sheets): gpt-5.5 high=0.836, gpt-5.5 low=0.813")
    for r in results:
        print(f"  {r['stem']:25} revR={r['revR']:.3f}")


if __name__ == "__main__":
    main()
