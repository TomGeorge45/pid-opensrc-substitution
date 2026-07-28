"""
POC runner v3 — Arm P (GPT-5.5-low), extends v2 with real Stage 13 (entity
validation) + Stage 12 (relation validation) via the REAL, unmodified
`stage_13_run` / `stage_12_run` drivers.

v2 (poc_run_arm_p_v2.py) stops after `build_relations` and scores entity/
relation F1 against PID2Graph ground truth. v3 does everything v2 does
through `build_relations`, then:

  1. Packs the resulting entities+relations into a real `RiveOntology`
     object (pnid_agent.models.rive_ontology) and writes it to
     `stage-11/rive_ontology.json` in the artifact store - the exact
     upstream artifact both stage_13_run and stage_12_run read.
  2. Calls the REAL `stage_13_run(...)` with `vlm_runner=RealOpenAIRunner()`
     (src/e2e_bench/assembly/real_openai_client.py - a drop-in swap for
     fake_llm.FakeRunner that talks to a live GPT-5.5-low Responses-API call
     instead of a canned answer) and `schema_factory=lambda: schemas` to
     bypass the tenant-ontology fetch (same schemas v2 already built).
  3. Re-scores entity F1 against the SAME ground truth using stage-13's
     output rive_ontology (post-validation entity set).
  4. Calls the REAL `stage_12_run(...)` the same way, reading from
     stage-13's output (matches prod: "Stage 12 reads from stage-13/ rather
     than stage-11/ so every relation is built on the finalized entity set").
  5. Re-scores relation F1 against the SAME ground truth using stage-12's
     output rive_ontology (post-validation relation set).

Reports entity/relation F1 at three checkpoints: pre-13/12 (== v2's number),
post-13, post-12.

Not run through the tenant ontology cache (schema_factory / ontology_payload_factory
bypass it, same as v2) - this is a deliberate scope choice consistent with the
rest of this benchmark harness (no live tenant fetch).
"""
import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from e2e_bench.assembly.document import build_artifact_store, build_single_page_document
from e2e_bench.assembly.entities import detections_to_entities
from e2e_bench.assembly.real_openai_client import RealOpenAIRunner
from e2e_bench.converters.stage01_classification import convert_classification
from e2e_bench.converters.stage04_detection import TileBatch, convert_detection
from e2e_bench.ontology import entity_type_names, load_benchmark_ontology_raw, load_ontology_relation_index
from e2e_bench.types import NormalizedDetection, NormalizedWord

from e2e_harness.graph_matcher import match_entities, match_relations
from e2e_harness.ground_truth import equipment_only, parse_graphml_ground_truth
from e2e_harness.holdout import holdout_sheet_paths
from e2e_harness.poc_run_arm_p_v2 import (
    anthropic_tool_to_openai,
    build_benchmark_schemas,
    gpt_detect_tile,
    payload_to_detections,
    run_paddle_ocr,
    serialize_tile_words,
)

from pnid_agent.models.page_classification import PageClassificationLabel
from pnid_agent.models.rive_ontology import DrawingMetadata, RiveOntology
from pnid_agent.stages.graph_construction.relations import build_relations
from pnid_agent.stages.line_tracing.driver import stage_06_run
from pnid_agent.sub_agents.entity_validation.driver import stage_13_run
from pnid_agent.sub_agents.relation_validation.driver import stage_12_run
from pnid_agent.sub_agents.symbol_detection.ontology_render import (
    build_detection_tool_schema,
    render_pid_ontology_for_prompt,
)
from pnid_agent.sub_agents.symbol_detection.prompt import build_system_prompt
from pnid_agent.sub_agents.symbol_detection.tile_words import slice_words_to_tile
from pnid_agent.stages.tile_segmentation.grid import compute_tile_grid

Image.MAX_IMAGE_PIXELS = None


def score(entities, relations, gt_equip, gt_edges, label: str) -> dict:
    em = match_entities(entities, gt_equip)
    rm = match_relations(relations, gt_edges, em)
    result = {
        "checkpoint": label,
        "n_entities": len(entities), "n_relations": len(relations),
        "entity_precision": em.precision, "entity_recall": em.recall, "entity_f1": em.f1,
        "relation_precision": rm.precision, "relation_recall": rm.recall, "relation_f1": rm.f1,
    }
    print(f"  [{label}] entity P={em.precision:.3f} R={em.recall:.3f} F1={em.f1:.3f} | "
          f"relation P={rm.precision:.3f} R={rm.recall:.3f} F1={rm.f1:.3f}")
    return result


def rive_ontology_from_bundle(entities, relations) -> RiveOntology:
    """Pack the entities/relations this harness already produces (real
    BundleEntity/BundleRelation objects from detections_to_entities /
    build_relations) into a real RiveOntology. Fields not relevant to this
    POC (service_line_legend, entity_class_legend, notes_index) are left at
    their pydantic defaults (empty dict) - all of RiveOntology's other fields
    are optional except ontology_version and drawing, both supplied below."""
    return RiveOntology(
        ontology_version="benchmark-v1",
        drawing=DrawingMetadata(),
        entities=entities,
        relations=relations,
    )


async def main(sheet_id: str, graphml_path: str, png_path: str):
    print(f"=== Arm P v3 (GPT-5.5-low, +stage13 entity validation, +stage12 relation validation) on {sheet_id} ===")
    store = build_artifact_store(tempfile.mkdtemp())
    doc = build_single_page_document(
        doc_id=sheet_id, job_id="job-armPv3-" + sheet_id, tenant_id="benchmark",
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
    # See poc_run_arm_p_v2.py's identical fix for why this substitution is needed.
    system_prompt = system_prompt.replace('entity_type "connector"', 'entity_type "inlet_outlet"')
    anthropic_tool_schema = build_detection_tool_schema(schemas)
    openai_tool = anthropic_tool_to_openai(anthropic_tool_schema)

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

    results = {}
    results["pre_13_12"] = score(entities, relations, gt_equip, gt_edges, "pre-13/12 (== v2)")

    # ── Pack + write stage-11/rive_ontology.json (the real upstream artifact
    # both stage_13_run and stage_12_run read) ──────────────────────────────
    rive = rive_ontology_from_bundle(entities, relations)
    store.write_json(doc.job_id, "stage-11/rive_ontology.json", rive.ui_payload())
    print(f"  wrote stage-11/rive_ontology.json: {len(rive.entities)} entities, {len(rive.relations)} relations")

    # ── Real Stage 13 (entity validation) via REAL GPT-5.5-low calls ───────
    print("  running REAL stage_13_run (entity validation, GPT-5.5-low)...")
    runner13 = RealOpenAIRunner(model="gpt-5.5", reasoning_effort="low")
    await stage_13_run(
        context, store, drawing_document=doc,
        schema_factory=lambda: schemas,
        vlm_runner=runner13,
        model="gpt-5.5-low",
    )
    s13_payload = store.read_json(doc.job_id, "stage-13/rive_ontology.json")
    rive_13 = RiveOntology.model_validate(s13_payload)
    print(f"  stage13 done: {len(rive_13.entities)} entities, {len(rive_13.relations)} relations after validation")
    results["post_13"] = score(rive_13.entities, rive_13.relations, gt_equip, gt_edges, "post-13")

    # ── Real Stage 12 (relation validation) via REAL GPT-5.5-low calls ─────
    print("  running REAL stage_12_run (relation validation, GPT-5.5-low)...")
    runner12 = RealOpenAIRunner(model="gpt-5.5", reasoning_effort="low")
    raw_ontology = load_benchmark_ontology_raw()
    await stage_12_run(
        context, store, drawing_document=doc,
        ontology_payload_factory=lambda: {"relations": raw_ontology["relations"]},
        vlm_runner=runner12,
        model="gpt-5.5-low",
        source_rive_uri="stage-13/rive_ontology.json",
    )
    s12_payload = store.read_json(doc.job_id, "stage-12/rive_ontology.json")
    rive_12 = RiveOntology.model_validate(s12_payload)
    print(f"  stage12 done: {len(rive_12.entities)} entities, {len(rive_12.relations)} relations after validation")
    results["post_12"] = score(rive_12.entities, rive_12.relations, gt_equip, gt_edges, "post-12")

    result = {
        "sheet_id": sheet_id, "arm": "gpt-5.5-low-v3-realprompt-paddleocr-stage13-stage12",
        "n_gt_entities": len(gt_equip), "n_gt_edges": len(gt_edges),
        "checkpoints": results,
    }
    print(json.dumps(result, indent=2))
    return result


async def main_all_sheets():
    sheets = holdout_sheet_paths()
    all_results = []
    for i, target in enumerate(sheets):
        print(f"\n{'='*70}\nSHEET {i+1}/{len(sheets)}: {target['sheet_id']}\n{'='*70}")
        try:
            r = await main(target["sheet_id"], target["graphml_path"], target["png_path"])
        except Exception as e:
            print(f"  [sheet-fail] {target['sheet_id']}: {type(e).__name__}: {e}")
            r = {"sheet_id": target["sheet_id"], "error": f"{type(e).__name__}: {e}"}
        all_results.append(r)

    out_path = "/tmp/arm_p_v3_all_sheets.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}\nALL SHEETS DONE - written to {out_path}\n{'='*70}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['sheet_id']}: ERROR - {r['error']}")
            continue
        c = r["checkpoints"]
        print(f"  {r['sheet_id']}: pre13/12 entityF1={c['pre_13_12']['entity_f1']:.3f}/relF1={c['pre_13_12']['relation_f1']:.3f}"
              f"  post13 entityF1={c['post_13']['entity_f1']:.3f}/relF1={c['post_13']['relation_f1']:.3f}"
              f"  post12 entityF1={c['post_12']['entity_f1']:.3f}/relF1={c['post_12']['relation_f1']:.3f}")
    return all_results


if __name__ == "__main__":
    asyncio.run(main_all_sheets())
