"""Round-trip tests (Conversion_Layer_Plan.md §7.1): every converter's output must
re-validate through the REAL agent's own pydantic model with zero errors."""
from pnid_agent.models.detections import Stage04Output
from pnid_agent.models.page_classification import PageClassificationLabel, Stage01Output
from pnid_agent.models.page_ocr import Stage015Output
from pnid_agent.models.title_block import Stage02Output

from e2e_bench.converters.stage01_classification import convert_classification
from e2e_bench.converters.stage015_ocr import convert_ocr, word_span_id
from e2e_bench.converters.stage02_titleblock import convert_titleblock
from e2e_bench.converters.stage04_detection import TileBatch, convert_detection
from e2e_bench.types import NormalizedDetection, NormalizedTitleBlock, NormalizedWord


def test_stage01_pid_drawing_roundtrip(doc, store):
    out = convert_classification(
        drawing_document=doc, artifact_store=store, page_index=0,
        classification=PageClassificationLabel.PID_DRAWING, confidence=0.9,
        model_version="qwen3-vl",
    )
    assert out.pages_to_process == [0]
    assert doc.pages[0].page_classification == PageClassificationLabel.PID_DRAWING
    raw = store.read_json(doc.job_id, "stage-01/stage_01_output.json")
    Stage01Output.model_validate(raw)


def test_stage01_other_classification_not_processed(doc, store):
    out = convert_classification(
        drawing_document=doc, artifact_store=store, page_index=0,
        classification=PageClassificationLabel.COVER, confidence=0.9, model_version="qwen3-vl",
    )
    assert out.pages_to_process == []
    assert out.pages_skipped == [0]


def test_stage015_ocr_roundtrip_and_word_order(doc, store):
    words = [
        NormalizedWord(text="VALVE-101", bbox=[10, 10, 50, 30], confidence=0.98),
        NormalizedWord(text="TK-200", bbox=[60, 60, 110, 80], confidence=0.95),
    ]
    out = convert_ocr(
        drawing_document=doc, artifact_store=store, page_index=0, words=words,
        engine_name="paddleocr", model_version="pp-ocrv4",
    )
    assert out.records[0].n_words == 2
    raw = store.read_json(doc.job_id, "stage-01.5/stage_01_5_output.json")
    Stage015Output.model_validate(raw)

    # word order is load-bearing (span_id indexes into this list)
    word_list = store.read_json(doc.job_id, "stage-01.5/intermediate/p0_words.json")
    assert [w["text"] for w in word_list] == ["VALVE-101", "TK-200"]
    assert word_span_id(0, 1) == "p0_w1"


def test_stage02_titleblock_located_false_is_safe(doc, store):
    ntb = NormalizedTitleBlock(located=False, fields={k: None for k in
                              ["drawing_number", "revision", "title", "site"]})
    out = convert_titleblock(
        drawing_document=doc, artifact_store=store, page_index=0, normalized=ntb,
        ocr_engine="paddleocr", source="e2e_bench_qwen",
    )
    assert out.pages_without_title_block == [0]
    raw = store.read_json(doc.job_id, "stage-02/title_block_output.json")
    Stage02Output.model_validate(raw)


def test_stage02_titleblock_xyxy_to_xywh_conversion(doc, store):
    ntb = NormalizedTitleBlock(
        located=True,
        fields={"drawing_number": "PIP-01-101", "revision": None, "title": "Test", "site": "X"},
        bbox_drawing_xyxy=[100, 100, 300, 200],
    )
    out = convert_titleblock(
        drawing_document=doc, artifact_store=store, page_index=0, normalized=ntb,
        ocr_engine="paddleocr", source="e2e_bench_qwen",
    )
    assert out.records[0].bbox_drawing == [100, 100, 200, 100]  # xywh: w=300-100, h=200-100
    raw = store.read_json(doc.job_id, "stage-02/title_block_output.json")
    Stage02Output.model_validate(raw)


def test_stage04_detection_upscale_and_ordering(doc, store):
    # tile 0: no upscale; tile 1: upscale 2 (Molmo2 512-config style)
    batch0 = TileBatch(tile_index=0, origin_xy=(0, 0), upscale=1.0, detections=[
        NormalizedDetection(bbox_tile=[100, 100, 140, 140], confidence=0.9, entity_type="valve"),
        NormalizedDetection(bbox_tile=[300, 300, 340, 340], confidence=0.5, entity_type="pump"),
    ])
    batch1 = TileBatch(tile_index=1, origin_xy=(1024, 0), upscale=2.0, detections=[
        NormalizedDetection(bbox_tile=[200, 200, 280, 280], confidence=0.8, entity_type="tank"),
    ])
    out, dropped = convert_detection(
        drawing_document=doc, artifact_store=store, page_index=0,
        tile_batches=[batch0, batch1], ocr_words_for_page=[], model_version="qwen3-vl-8b-zeroshot",
    )
    assert dropped == []
    dets = out.pages[0].detections
    assert len(dets) == 3
    # upscale math: [200,200,280,280] / 2 + (1024,0) origin -> [1124,100,1164,140]
    tank = next(d for d in dets if d.entity_type == "tank")
    assert tank.provenance.bbox == [1124, 100, 1164, 140]
    # (y0,x0) ordering
    assert [d.detection_id for d in dets] == ["p0_d000", "p0_d001", "p0_d002"]
    raw = store.read_json(doc.job_id, "stage-04/stage_04_output.json")
    Stage04Output.model_validate(raw)


def test_stage04_never_drops_nonzero_confidence(doc, store):
    batch0 = TileBatch(tile_index=0, origin_xy=(0, 0), upscale=1.0, detections=[
        NormalizedDetection(bbox_tile=[10, 10, 50, 50], confidence=0.01, entity_type="valve"),
    ])
    out, dropped = convert_detection(
        drawing_document=doc, artifact_store=store, page_index=0,
        tile_batches=[batch0], ocr_words_for_page=[], model_version="qwen3-vl-8b-zeroshot",
    )
    assert len(out.pages[0].detections) == 1  # low but nonzero confidence survives
