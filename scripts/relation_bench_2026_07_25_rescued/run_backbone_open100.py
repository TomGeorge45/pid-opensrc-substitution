"""Multi-sheet real benchmark for the gap #14 backbone pass -- all 12 real OPEN100 sheets
(the genuine real-world PID2Graph subset). Dataset PID's 500 synthetic sheets are excluded
from THIS run: they're ~7168x4561px (7x OPEN100's pixel count) and R1's raster CV pipeline
did not finish a single one within 5+ minutes -- a real scaling limitation to report
honestly, not something to force through with a longer timeout.
"""
import sys
import time

sys.path.insert(0, "/Users/tomgeorge/pid-ml/src/relation_bench")

from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pid2graph_gt import parse_graphml, contract
from line_tracing import process_page
from graph_construction import build_topology_relations, build_junction_to_detection_map
from score import score_topology, aggregate, format_report

ROOT = Path("/System/Volumes/Data/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6/scratchpad/molmo_ft_prep_v2/pid2graph/PID2Graph/Complete/PID2Graph OPEN100")
sheets = sorted(ROOT.glob("*.graphml"))
print(f"running {len(sheets)} real OPEN100 sheets", flush=True)

before_results = []
after_results = []
t_start = time.monotonic()

for i, graphml_path in enumerate(sheets):
    png_path = graphml_path.with_suffix(".png")
    if not png_path.exists():
        print(f"[{i+1}/{len(sheets)}] {graphml_path.stem}: SKIPPED (no png)", flush=True)
        continue

    nodes, raw_edges, _ = parse_graphml(graphml_path)
    gt = contract(graphml_path.stem, nodes, raw_edges)
    if not gt.nodes:
        print(f"[{i+1}/{len(sheets)}] {graphml_path.stem}: SKIPPED (no symbol nodes)", flush=True)
        continue

    img = Image.open(png_path).convert("L")
    page_gray = np.array(img)
    ids = list(gt.nodes.keys())
    bboxes = [tuple(int(v) for v in gt.nodes[nid].bbox) for nid in ids]

    t_sheet = time.monotonic()
    page_graph = process_page(page_index=0, page_gray=page_gray, symbol_bboxes=bboxes, symbol_det_ids=ids)
    j2d = build_junction_to_detection_map(page_graph.junctions, bboxes, ids)

    before = build_topology_relations(page_graph, ids, junction_to_detection_id=j2d)
    before_score = score_topology(gt, {r.pair for r in before}, sheet_group="sparse")
    before_results.append(before_score)

    symbol_class_of = {nid: gt.nodes[nid].cls for nid in ids}
    passthrough = {"valve", "instrumentation", "pump"}
    after = build_topology_relations(
        page_graph, ids, junction_to_detection_id=j2d,
        symbol_class_of=symbol_class_of, passthrough_symbol_classes=passthrough)
    after_score = score_topology(gt, {r.pair for r in after}, sheet_group="sparse")
    after_results.append(after_score)

    dt = time.monotonic() - t_sheet
    print(f"[{i+1}/{len(sheets)}] {graphml_path.stem}: {len(gt.nodes)} nodes, "
         f"{len(gt.edge_pairs)} gt edges -- F1 {before_score.f1:.3f} -> {after_score.f1:.3f} "
         f"({dt:.1f}s)", flush=True)

print(f"\ntotal elapsed: {time.monotonic()-t_start:.0f}s\n", flush=True)

print("=" * 70, flush=True)
print("AGGREGATE -- BEFORE (no backbone pass), 12 real OPEN100 sheets", flush=True)
print("=" * 70, flush=True)
agg_before = aggregate(before_results)
print(format_report("R1+R2a", agg_before), flush=True)

print("\n" + "=" * 70, flush=True)
print("AGGREGATE -- AFTER (backbone pass enabled), 12 real OPEN100 sheets", flush=True)
print("=" * 70, flush=True)
agg_after = aggregate(after_results)
print(format_report("R1+R2a+backbone", agg_after), flush=True)

print(f"\n\nDELTA: F1 {agg_before.f1:.3f} -> {agg_after.f1:.3f}  "
     f"(tp {agg_before.tp}->{agg_after.tp}, fp {agg_before.fp}->{agg_after.fp}, "
     f"fn {agg_before.fn}->{agg_after.fn})", flush=True)
