"""
POC runner v2 — Arm P (GPT-5.5-low), fidelity rebuild.

v1 (poc_run_arm_p.py) used an ad hoc 10-line prompt with no OCR words, which
understated real capability: the actual production stage 4 prompt (916 lines,
prompt.py) is fed a Stage 1.5 OCR word list per tile and explicitly told to
read tags from THAT LIST, not from the image. v1 skipped OCR entirely and
asked the model to read tag text straight from pixels - a harder task than
production actually sets. v1 result on OPEN100/8: entity F1 0.414, relation F1
0.0, confounded by ~75% of detections getting no derivable value.

v2 fixes the confound by reusing REAL production code unmodified:
  - pnid_agent.sub_agents.symbol_detection.prompt.build_system_prompt (the
    real 916-line frame + ISA reference)
  - pnid_agent.sub_agents.symbol_detection.ontology_render.build_detection_tool_schema
    (the real detect_symbols Anthropic tool schema, entity_type enum-constrained)
  - pnid_agent.sub_agents.symbol_detection.tile_words.slice_words_to_tile (the
    real per-tile OCR word slicing, same margin_px=24 default)

Real prod Stage 1.5 OCR = Google Cloud Vision (needs GOOGLE_CLOUD_VISION_API_KEY,
which this repo doesn't have). Per user decision 2026-07-16: substitute PaddleOCR
here instead, since it's this project's own decided local OCR substitute anyway -
this measures "GPT-5.5 detection w/ real prompt + our real local OCR substitute,"
not literally byte-for-byte prod, and that's an accepted, explicit tradeoff.

The Anthropic tool schema is translated to OpenAI function-calling format
(input_schema -> parameters) since we call OpenAI's API directly rather than
through prod's Anthropic-format LiteLLM proxy - purely a wire-format translation,
not a prompt/schema content change.
"""
import asyncio
import base64
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from e2e_bench.assembly.document import build_artifact_store, build_single_page_document
from e2e_bench.assembly.entities import detections_to_entities
from e2e_bench.converters.stage01_classification import convert_classification
from e2e_bench.converters.stage04_detection import TileBatch, convert_detection
from e2e_bench.ontology import entity_type_names, load_ontology_relation_index
from e2e_bench.types import NormalizedDetection

from e2e_harness.graph_matcher import match_entities, match_relations
from e2e_harness.ground_truth import equipment_only, parse_graphml_ground_truth
from e2e_harness.holdout import holdout_sheet_paths

from pnid_agent.models.page_classification import PageClassificationLabel
from pnid_agent.models.page_ocr import OcrWord
from pnid_agent.stages.graph_construction.relations import build_relations
from pnid_agent.stages.line_tracing.driver import stage_06_run
from pnid_agent.sub_agents.symbol_detection.ontology_render import (
    build_detection_tool_schema,
    render_pid_ontology_for_prompt,
)
from pnid_agent.sub_agents.symbol_detection.prompt import build_system_prompt
from pnid_agent.sub_agents.symbol_detection.tile_words import slice_words_to_tile
from pnid_agent.sub_agents.title_block_extraction.ontology import EntityExtractionSchema
from pnid_agent.stages.tile_segmentation.grid import compute_tile_grid

Image.MAX_IMAGE_PIXELS = None
_oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class _EmptyAttrs(BaseModel):
    pass


def build_benchmark_schemas():
    """EntityExtractionSchema stand-ins for the 7-type benchmark ontology - no
    attributes (our ontology carries none), matching what render_pid_ontology_for_prompt
    and build_detection_tool_schema need."""
    return [
        EntityExtractionSchema(
            entity_type=k, model=_EmptyAttrs, attribute_metadata={},
            raw_sample_payload={"descriptions": {"en": v}},
        )
        for k, v in entity_type_names().items()
    ]


def anthropic_tool_to_openai(tool_schema: dict) -> dict:
    """Responses-API function-tool shape (flat, not chat.completions' nested
    {"type":"function","function":{...}}) - chat.completions rejects function
    tools combined with reasoning_effort != "none" for gpt-5.5; /v1/responses
    supports tools + reasoning_effort="low" together, which is closer to the
    real GPT-5.5-low config than degrading to reasoning_effort="none"."""
    return {
        "type": "function",
        "name": tool_schema["name"],
        "description": tool_schema["description"],
        "parameters": tool_schema["input_schema"],
        "strict": False,
    }


def run_paddle_ocr(png_path: str):
    """Real production stage 1.5 substitute - PaddleOCR full-page pass, mapped
    to the real OcrWord dataclass (page coords, matching Stage 1.5's contract).

    PaddleOCR 3.7.0 (actually installed) dropped the classic `.ocr()` list-of-lines
    shape parse_paddle.py's `parse_paddle_ocr_result` targets - `.ocr()` now just
    proxies to the new `.predict()` pipeline and returns one `OCRResult` dict per
    page with `rec_texts`/`rec_scores`/`rec_boxes` parallel arrays (axis-aligned
    [x0,y0,x1,y1], already in original-image pixel coords). Parsed directly here;
    `parse_paddle_ocr_result` is left as-is for whichever PaddleOCR version the
    eventual Arm L build standardizes on - flag this version skew there too."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en")
    result = ocr.predict(png_path)
    if not result:
        return []
    page = result[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    boxes = page.get("rec_boxes", [])
    words = []
    for text, score, box in zip(texts, scores, boxes):
        bbox = [int(round(v)) for v in box]
        words.append(OcrWord(text=text, bbox=bbox, confidence=float(score)))
    return words


def serialize_tile_words(words_in_tile, *, page_index: int) -> str:
    """Mirrors detector.py's private _serialize_tile_words - same [span_id] 'text'
    bbox=[...] format the real prompt's user message uses."""
    if not words_in_tile:
        return "  (none)"
    lines = []
    for w in words_in_tile[:2500]:
        span_id = f"p{page_index}_w{w.global_idx}"
        lines.append(f"  [{span_id}] {w.text!r:40}  bbox={w.bbox}")
    return "\n".join(lines)


def gpt_detect_tile(system_prompt, tile_image, tile_id, tile_origin, words_in_tile,
                     openai_tool, page_index, max_tokens=8192):
    buf = io.BytesIO(); tile_image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    words_dump = serialize_tile_words(words_in_tile, page_index=page_index)
    user_text = (
        f"Tile id: {tile_id}\n"
        f"Tile drawing-coord origin: x={tile_origin[0]}, y={tile_origin[1]} (top-left of the tile in PAGE coords).\n\n"
        "Stage 1.5 OCR words on (or near) this tile. Each line is "
        "``[span_id] 'text' bbox=[x0, y0, x1, y1]`` in ORIGINAL page coords. "
        "Use the FULL span_id verbatim in your associations:\n"
        f"{words_dump}\n\n"
        "Detect every symbol on the tile and associate tags by span_id. Return tile-local bboxes only."
    )
    try:
        resp = _oa.responses.create(
            model="gpt-5.5", reasoning={"effort": "low"}, max_output_tokens=max_tokens,
            tools=[openai_tool],
            tool_choice={"type": "function", "name": "detect_symbols"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                    {"type": "input_text", "text": user_text},
                ]},
            ],
        )
        call = next(item for item in resp.output if item.type == "function_call")
        return json.loads(call.arguments)
    except Exception as e:
        print(f"  [api-fail] tile {tile_id}: {type(e).__name__}: {e}")
        return {"symbols": [], "associations": [], "unmapped_observations": []}


def payload_to_detections(payload: dict) -> list:
    valid_types = set(entity_type_names().keys())
    out = []
    for spec in payload.get("symbols") or []:
        if not isinstance(spec, dict):
            continue
        entity_type = spec.get("entity_type")
        bbox = spec.get("bbox_tile")
        if entity_type not in valid_types or not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        value = spec.get("value") or spec.get("name") or None
        out.append(NormalizedDetection(
            bbox_tile=[float(v) for v in bbox], confidence=float(spec.get("confidence", 0.7)),
            entity_type=entity_type, value=value,
            entity_subtype=spec.get("entity_subtype"), description=spec.get("description"),
        ))
    return out


async def main(sheet_id: str, graphml_path: str, png_path: str):
    print(f"=== Arm P v2 (GPT-5.5-low, real prompt+schema, PaddleOCR) on {sheet_id} ===")
    store = build_artifact_store(tempfile.mkdtemp())
    doc = build_single_page_document(
        doc_id=sheet_id, job_id="job-armPv2-" + sheet_id, tenant_id="benchmark",
        image_path=png_path, artifact_store=store,
    )
    context = SimpleNamespace(tenant_id="benchmark")

    convert_classification(
        drawing_document=doc, artifact_store=store, page_index=0,
        classification=PageClassificationLabel.PID_DRAWING, confidence=1.0,
        model_version="gpt-5.5-low-assumed",
    )

    schemas = build_benchmark_schemas()
    ontology_rendering = render_pid_ontology_for_prompt(schemas)
    system_prompt = build_system_prompt(ontology_rendering)
    # 2026-07-16 diagnostic finding: the real ISA reference tells the model to emit
    # off-page connectors as entity_type "connector" - our benchmark ontology names
    # that type "inlet_outlet" instead. Since the tool schema enum-constrains
    # entity_type to our 7 real names, "connector" is impossible to emit and those
    # symbols were silently landing in unmapped_observations (~18% of GT entities on
    # sheet 8 - the off-page pennant/flag boxes). Substitute the ontology's real name
    # into the reused prompt text rather than editing prod's prompt.py.
    system_prompt = system_prompt.replace('entity_type "connector"', 'entity_type "inlet_outlet"')
    anthropic_tool_schema = build_detection_tool_schema(schemas)
    openai_tool = anthropic_tool_to_openai(anthropic_tool_schema)
    print(f"  real system prompt: {len(system_prompt)} chars; entity_type enum: "
          f"{anthropic_tool_schema['input_schema']['properties']['symbols']['items']['properties']['entity_type']['enum']}")

    print("  running PaddleOCR (stage 1.5 substitute)...")
    page_words = run_paddle_ocr(png_path)
    print(f"  {len(page_words)} OCR words")

    full_img = Image.open(png_path).convert("RGB")
    W, H = full_img.size
    tiles = compute_tile_grid(drawing_bbox=[0, 0, W, H], page_size=(W, H))
    print(f"  {len(tiles)} tiles (1024/205 grid)")

    tile_batches = []
    for t in tiles:
        crop = full_img.crop((t.x0, t.y0, t.x1, t.y1))
        words_in_tile = slice_words_to_tile(page_words, tile_bbox=[t.x0, t.y0, t.x1, t.y1], margin_px=24)
        tile_id = f"p0_t{t.idx:03d}"
        payload = gpt_detect_tile(system_prompt, crop, tile_id, (t.x0, t.y0), words_in_tile,
                                  openai_tool, page_index=0)
        dets = payload_to_detections(payload)
        print(f"    tile {t.idx} ({t.x0},{t.y0})-({t.x1},{t.y1}): {len(words_in_tile)} ocr words, {len(dets)} detections")
        tile_batches.append(TileBatch(tile_index=t.idx, origin_xy=(t.x0, t.y0), upscale=1.0, detections=dets))

    from e2e_bench.types import NormalizedWord
    normalized_ocr_words = [NormalizedWord(text=w.text, bbox=w.bbox, confidence=w.confidence) for w in page_words]

    s4_out, dropped = convert_detection(
        drawing_document=doc, artifact_store=store, page_index=0,
        tile_batches=tile_batches, ocr_words_for_page=normalized_ocr_words, model_version="gpt-5.5-low",
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
    print(f"  entities built: {len(entities)} (of {len(s4_out.pages[0].detections)} detections)")

    ontology_idx = load_ontology_relation_index()
    relations, _meta, unresolved = build_relations(s6_out.pages[0], det_to_temp, type_by_temp, ontology_idx)
    print(f"  relations built: {len(relations)}, unresolved: {len(unresolved)}")

    gt_entities, gt_edges = parse_graphml_ground_truth(graphml_path)
    gt_equip = equipment_only(gt_entities)
    print(f"  GT: {len(gt_equip)} equipment entities, {len(gt_edges)} edges")

    em = match_entities(entities, gt_equip)
    rm = match_relations(relations, gt_edges, em)

    # Diagnostic dump for overlay inspection (2026-07-16: entity recall bottleneck
    # investigation) - not part of the scoring path, just a debug artifact.
    diag_path = f"/tmp/{sheet_id}_diag.json"
    with open(diag_path, "w") as f:
        json.dump({
            "ocr_words": [{"text": w.text, "bbox": w.bbox, "confidence": w.confidence} for w in page_words],
            "entities": [{"entity_type": e.entity_type, "bbox": e.source_bbox, "name": getattr(e, "name", None)}
                        for e in entities],
            "gt_equip": [{"node_id": g.node_id, "label": g.label, "bbox": g.bbox} for g in gt_equip],
            "page_size": [W, H],
        }, f, indent=2)
    print(f"  diagnostic dump: {diag_path}")

    result = {
        "sheet_id": sheet_id, "arm": "gpt-5.5-low-v2-realprompt-paddleocr",
        "n_ocr_words": len(page_words), "n_tiles": len(tiles),
        "n_detections_pre_nms": sum(len(b.detections) for b in tile_batches),
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
    target = sheets[0]
    asyncio.run(main(target["sheet_id"], target["graphml_path"], target["png_path"]))
