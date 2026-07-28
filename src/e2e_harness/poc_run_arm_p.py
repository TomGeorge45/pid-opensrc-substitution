"""
POC runner — Arm P (GPT-5.5-low), one sheet, base graph only (stage 1 -> 4 -> 6 -> entity
assembly -> relations -> score). Stage 13/12 validation deliberately deferred to a fast
follow-up once this base pipeline is proven (E2E_Harness_Plan.md POC scope trim).

Known POC simplification, documented not hidden: stage 1.5 OCR is skipped (empty word
list) for this first pass - PaddleOCR (Arm L's real substitute) isn't benchmarked yet
either, and an empty OCR word list is a supported, already-tested path through the
conversion layer (stage04_detection's OCR-anchor-correction just doesn't fire). Title
block (stage 2) is also skipped for this pass - PID2Graph sheets have no title-block
ground truth to check it against anyway, and it doesn't affect entity/relation F1.
"""
import asyncio
import io
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace

from openai import OpenAI
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from e2e_bench.assembly.document import build_artifact_store, build_single_page_document
from e2e_bench.assembly.entities import detections_to_entities
from e2e_bench.converters.stage01_classification import convert_classification
from e2e_bench.converters.stage04_detection import TileBatch, convert_detection
from e2e_bench.ontology import entity_types, load_ontology_relation_index
from e2e_bench.types import NormalizedDetection

from e2e_harness.graph_matcher import match_entities, match_relations
from e2e_harness.ground_truth import equipment_only, parse_graphml_ground_truth
from e2e_harness.holdout import holdout_sheet_paths

from pnid_agent.models.page_classification import PageClassificationLabel
from pnid_agent.stages.graph_construction.relations import build_relations
from pnid_agent.stages.line_tracing.driver import stage_06_run
from pnid_agent.stages.tile_segmentation.grid import compute_tile_grid

Image.MAX_IMAGE_PIXELS = None

_oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ENTITY_TYPES = entity_types()

DETECTION_PROMPT = """You are an expert P&ID reviewer analyzing one tile cropped from a larger Piping & Instrumentation Diagram. The image is {W}x{H} pixels.

Find every discrete equipment symbol in this tile: valves, instruments/gauges, pumps, tanks/vessels, inlet/outlet arrows, and any other general equipment icon. Do NOT count plain pipe/line segments, text labels alone, or dimension arrows.

For each symbol found, classify it as exactly one of: {types}.

If the symbol has a visible tag/label text near it (e.g. "V-101", "PT-200"), include that as "value". If no legible tag, use null.

Output your final answer as a JSON array inside a ```json code fence. Each item: [x0, y0, x1, y1, "entity_type", "value_or_null"] - tight box around just the icon, absolute pixel coordinates in the {W}x{H} tile image (top-left origin). If no symbols: []"""


def gpt_generate(image, prompt, max_tokens=3000):
    buf = io.BytesIO(); image.save(buf, format="PNG")
    b64 = __import__("base64").standard_b64encode(buf.getvalue()).decode()
    try:
        resp = _oa.chat.completions.create(
            model="gpt-5.5", reasoning_effort="low", max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}])
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [api-fail] {type(e).__name__}: {e}")
        return ""


def parse_detection_json(text, tile_w, tile_h):
    fenced = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    candidate = fenced[-1] if fenced else text
    try:
        items = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    out = []
    for item in items:
        if not (isinstance(item, list) and len(item) >= 5):
            continue
        x0, y0, x1, y1 = (float(v) for v in item[:4])
        entity_type = item[4] if item[4] in ENTITY_TYPES else "general"
        value = item[5] if len(item) > 5 and item[5] else None
        out.append(NormalizedDetection(bbox_tile=[x0, y0, x1, y1], confidence=0.7,
                                       entity_type=entity_type, value=value))
    return out


async def main(sheet_id: str, graphml_path: str, png_path: str):
    print(f"=== Arm P (GPT-5.5-low) on {sheet_id} ===")
    store = build_artifact_store(tempfile.mkdtemp())
    doc = build_single_page_document(
        doc_id=sheet_id, job_id="job-armP-" + sheet_id, tenant_id="benchmark",
        image_path=png_path, artifact_store=store,
    )
    context = SimpleNamespace(tenant_id="benchmark")

    convert_classification(
        drawing_document=doc, artifact_store=store, page_index=0,
        classification=PageClassificationLabel.PID_DRAWING, confidence=1.0,
        model_version="gpt-5.5-low-assumed",  # POC: real sheets are known P&IDs, stage-1 call skipped for speed
    )

    full_img = Image.open(png_path).convert("RGB")
    W, H = full_img.size
    tiles = compute_tile_grid(drawing_bbox=[0, 0, W, H], page_size=(W, H))
    print(f"  {len(tiles)} tiles (1024/205 grid)")

    tile_batches = []
    for t in tiles:
        crop = full_img.crop((t.x0, t.y0, t.x1, t.y1))
        tw, th = crop.size
        raw = gpt_generate(crop, DETECTION_PROMPT.format(W=tw, H=th, types=", ".join(ENTITY_TYPES)))
        dets = parse_detection_json(raw, tw, th)
        print(f"    tile {t.idx} ({t.x0},{t.y0})-({t.x1},{t.y1}): {len(dets)} detections")
        tile_batches.append(TileBatch(tile_index=t.idx, origin_xy=(t.x0, t.y0), upscale=1.0, detections=dets))

    s4_out, dropped = convert_detection(
        drawing_document=doc, artifact_store=store, page_index=0,
        tile_batches=tile_batches, ocr_words_for_page=[], model_version="gpt-5.5-low",
    )
    print(f"  stage4: {len(s4_out.pages[0].detections)} detections after NMS, {len(dropped)} dropped in compose")

    await stage_06_run(context, store, drawing_document=doc)
    s6_raw = store.read_json(doc.job_id, "stage-06/stage_06_output.json")
    from pnid_agent.models.line_tracing import Stage06Output
    s6_out = Stage06Output.model_validate(s6_raw)
    print(f"  stage6: {len(s6_out.pages[0].segments)} segments")

    page_size = (W, H)
    entities, det_to_temp, type_by_temp = detections_to_entities(
        detections=s4_out.pages[0].detections, page_index=0, page_size=page_size,
        stage_4_model_version="gpt-5.5-low",
    )
    print(f"  entities built: {len(entities)} (of {len(s4_out.pages[0].detections)} detections - "
         f"drop = no derivable name/value, see Conversion_Layer_Plan.md discovery)")

    ontology_idx = load_ontology_relation_index()
    relations, _meta, unresolved = build_relations(s6_out.pages[0], det_to_temp, type_by_temp, ontology_idx)
    print(f"  relations built: {len(relations)}, unresolved: {len(unresolved)}")

    gt_entities, gt_edges = parse_graphml_ground_truth(graphml_path)
    gt_equip = equipment_only(gt_entities)
    print(f"  GT: {len(gt_equip)} equipment entities, {len(gt_edges)} edges")

    em = match_entities(entities, gt_equip)
    rm = match_relations(relations, gt_edges, em)

    result = {
        "sheet_id": sheet_id, "arm": "gpt-5.5-low",
        "n_tiles": len(tiles), "n_detections_pre_nms": sum(len(b.detections) for b in tile_batches),
        "n_detections_post_nms": len(s4_out.pages[0].detections),
        "n_entities": len(entities), "n_relations": len(relations),
        "n_gt_entities": len(gt_equip), "n_gt_edges": len(gt_edges),
        "entity_precision": em.precision, "entity_recall": em.recall, "entity_f1": em.f1,
        "relation_precision": rm.precision, "relation_recall": rm.recall, "relation_f1": rm.f1,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    sheets = holdout_sheet_paths()
    target = sheets[0]  # start with the smallest sheet (OPEN100_8) for a fast first result
    asyncio.run(main(target["sheet_id"], target["graphml_path"], target["png_path"]))
