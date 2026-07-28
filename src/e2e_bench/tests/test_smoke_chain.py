"""
Smoke chain test (Conversion_Layer_Plan.md §7.5): one real P&ID sheet, MOCK normalized
answers (no live model calls), run through every stage that matters for the end-to-end
benchmark — stage 1 -> stage 4 -> REAL stage_06_run -> entity assembly -> REAL
build_relations -> stage 10.5 sidecar -> REAL infer_from_skid_groups -> REAL stage_13_run
(via FakeMessagesClient) -> REAL stage_12_run (via FakeMessagesClient).

This is the test that proves the whole conversion layer holds together, not just each
piece in isolation. Every intermediate artifact is re-validated through the agent's own
pydantic models.
"""
import pytest

from e2e_bench.assembly.entities import detections_to_entities
from e2e_bench.converters.stage01_classification import convert_classification
from e2e_bench.converters.stage04_detection import TileBatch, convert_detection
from e2e_bench.converters.stage105_skid import convert_skid_groups
from e2e_bench.converters.stage12_relation_validation import convert_relation_validation
from e2e_bench.converters.stage13_entity_validation import convert_entity_validation
from e2e_bench.ontology import load_ontology_relation_index
from e2e_bench.types import (
    NormalizedDetection,
    NormalizedEntityVerdict,
    NormalizedRelationVerdict,
    NormalizedSkidAssignment,
    NormalizedSkidMember,
)

from pnid_agent.models.detections import Stage04Output
from pnid_agent.models.line_tracing import Stage06Output
from pnid_agent.models.page_classification import PageClassificationLabel
from pnid_agent.models.rive_ontology import BundleRelation, DrawingMetadata
from pnid_agent.models.rive_ontology import RiveOntology as RO
from pnid_agent.stages.graph_construction.inference import infer_from_skid_groups
from pnid_agent.stages.graph_construction.relations import build_relations
from pnid_agent.stages.line_tracing.driver import stage_06_run


@pytest.mark.asyncio
async def test_full_chain_one_sheet(doc, store, context):
    # ---- stage 1 ----
    convert_classification(
        drawing_document=doc, artifact_store=store, page_index=0,
        classification=PageClassificationLabel.PID_DRAWING, confidence=0.95,
        model_version="qwen3-vl",
    )
    assert doc.pages[0].page_classification == PageClassificationLabel.PID_DRAWING

    # ---- stage 4 (asset gets a value so build_entity doesn't drop it - see types.py's
    # NormalizedDetection docstring for why this is required, not optional) ----
    batch0 = TileBatch(tile_index=0, origin_xy=(0, 0), upscale=1.0, detections=[
        NormalizedDetection(bbox_tile=[60, 480, 100, 520], confidence=0.9,
                           entity_type="valve", value="V-101"),
        NormalizedDetection(bbox_tile=[720, 800, 900, 870], confidence=0.85,
                           entity_type="asset", value="TK-200"),
    ])
    s4_out, dropped = convert_detection(
        drawing_document=doc, artifact_store=store, page_index=0,
        tile_batches=[batch0], ocr_words_for_page=[], model_version="qwen3-vl-8b-zeroshot",
    )
    assert dropped == []
    assert len(s4_out.pages[0].detections) == 2
    Stage04Output.model_validate(store.read_json(doc.job_id, "stage-04/stage_04_output.json"))

    # ---- stage 6 (REAL driver, deterministic) ----
    await stage_06_run(context, store, drawing_document=doc)
    s6_out = Stage06Output.model_validate(
        store.read_json(doc.job_id, "stage-06/stage_06_output.json")
    )
    assert len(s6_out.pages) == 1
    assert len(s6_out.pages[0].segments) > 0  # real lines traced from the real image

    # ---- entity assembly ----
    page_size = (doc.pages[0].raster.width_px, doc.pages[0].raster.height_px)
    entities, det_to_temp, type_by_temp = detections_to_entities(
        detections=s4_out.pages[0].detections, page_index=0, page_size=page_size,
        stage_4_model_version="qwen3-vl-8b-zeroshot",
    )
    assert len(entities) == 2  # both survive because both have a `value`
    entities_by_temp = {e.temp_id: e for e in entities}

    # ---- stage 11 relations (REAL build_relations) ----
    ontology_idx = load_ontology_relation_index()
    relations, _metadata, unresolved = build_relations(
        s6_out.pages[0], det_to_temp, type_by_temp, ontology_idx,
    )
    # not asserting len(relations) > 0 - arbitrary bbox placement in this fixture image
    # may or may not have a traced line between them; the meaningful assertion is that
    # build_relations ran without error against real line-graph data.
    assert isinstance(relations, list)

    # ---- stage 10.5 skid sidecar + REAL infer_from_skid_groups ----
    asset_temp_id = next(e.temp_id for e in entities if e.entity_type == "asset")
    valve_temp_id = next(e.temp_id for e in entities if e.entity_type == "valve")
    assignment = NormalizedSkidAssignment(asset_temp_id=asset_temp_id, members=[
        NormalizedSkidMember(target_temp_id=valve_temp_id, forward_relation_name="Installed Valves",
                            confidence=0.9, reasoning="colocated"),
    ])
    sidecar = convert_skid_groups(
        drawing_document=doc, artifact_store=store, pages_processed=[0], assignments=[assignment],
    )
    skid_relations, _skid_meta, _next_pos = infer_from_skid_groups(
        entities, ontology_idx, sidecar["groups"], page_index=0, starting_relation_position=0,
    )
    assert len(skid_relations) == 1
    assert skid_relations[0].source_temp_id == asset_temp_id
    assert skid_relations[0].target_temp_id == valve_temp_id

    # ---- write stage-11 artifact (combining structural + skid relations) for stage 13/12 ----
    all_relations = list(relations) + list(skid_relations)
    rive = RO(ontology_version="e2e-bench-v1", drawing=DrawingMetadata(),
             entities=entities, relations=all_relations)
    store.write_json(doc.job_id, "stage-11/rive_ontology.json", rive.model_dump(mode="json"))

    # ---- stage 13 (REAL stage_13_run via FakeMessagesClient) ----
    verdicts = [
        NormalizedEntityVerdict(temp_id=valve_temp_id, keep=True, confidence=0.9, reasoning="real"),
        NormalizedEntityVerdict(temp_id=asset_temp_id, keep=False, confidence=0.9, reasoning="spurious"),
    ]
    await convert_entity_validation(
        context=context, artifact_store=store, drawing_document=doc, verdicts=verdicts,
    )
    s13_rive = RO.model_validate(store.read_json(doc.job_id, "stage-13/rive_ontology.json"))
    remaining_ids = {e.temp_id for e in s13_rive.entities}
    assert valve_temp_id in remaining_ids
    assert asset_temp_id not in remaining_ids
    # the skid relation touching the removed asset should be stamped rejected, not deleted
    # (real write semantics: entity_validation/driver.py:306-333)
    skid_rel_after = next(r for r in s13_rive.relations if r.relation_id == skid_relations[0].relation_id)
    assert skid_rel_after.review_status == "rejected"

    # ---- stage 12 (REAL stage_12_run via FakeMessagesClient), on a fresh low-confidence
    # relation so it's actually selected for validation ----
    extra_rel = BundleRelation(
        relation_id="p0_rextra", forward_relation_name="connects to",
        reverse_relation_name="connected from", source_entity_type="valve",
        source_temp_id=valve_temp_id, target_entity_type="valve", target_temp_id=valve_temp_id,
        confidence=0.5,
    )
    s13_rive.relations.append(extra_rel)
    store.write_json(doc.job_id, "stage-13/rive_ontology.json", s13_rive.model_dump(mode="json"))

    entities_by_temp_after = {e.temp_id: e for e in s13_rive.entities}
    verdict = NormalizedRelationVerdict(relation_id="p0_rextra", verdict="confirmed",
                                        revised_confidence=0.95, reasoning="looks right")
    await convert_relation_validation(
        context=context, artifact_store=store, drawing_document=doc, verdicts=[verdict],
        relations_by_id={"p0_rextra": extra_rel}, entities_by_temp_id=entities_by_temp_after,
        source_rive_uri="stage-13/rive_ontology.json",
    )
    s12_rive = RO.model_validate(store.read_json(doc.job_id, "stage-12/rive_ontology.json"))
    validated = next(r for r in s12_rive.relations if r.relation_id == "p0_rextra")
    assert validated.confidence == 0.95
