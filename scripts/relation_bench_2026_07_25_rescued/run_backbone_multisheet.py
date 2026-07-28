"""Multi-sheet real benchmark for the gap #14 backbone pass -- moves beyond the single
OPEN100/0 self-test to a real aggregate, stratified sparse (OPEN100) vs dense (Dataset PID)
per gap #17/#19 convention. R1 (raster tracer) + R2a, before/after the backbone pass.
"""
import random
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

ROOT = Path("/System/Volumes/Data/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6/scratchpad/molmo_ft_prep_v2/pid2graph/PID2Graph/Complete")

OPEN100 = sorted((ROOT / "PID2Graph OPEN100").glob("*.graphml"))
rng = random.Random(4242)
DATASET_PID_ALL = sorted((ROOT / "Dataset PID").glob("*.graphml"))
DATASET_PID_SAMPLE = rng.sample(DATASET_PID_ALL, 20)

sheets = [(g, "sparse") for g in OPEN100] + [(g, "dense") for g in DATASET_PID_SAMPLE]
print(f"running {len(sheets)} sheets ({len(OPEN100)} sparse/OPEN100 + {len(DATASET_PID_SAMPLE)} dense/Dataset-PID sample)\n")

before_results = []
after_results = []
skipped = []
t_start = time.monotonic()

for i, (graphml_path, group) in enumerate(sheets):
    png_path = graphml_path.with_suffix(".png")
    if not png_path.exists():
        skipped.append((graphml_path.stem, "no matching png"))
        continue
    try:
        nodes, raw_edges, _ = parse_graphml(graphml_path)
        gt = contract(f"{group}/{graphml_path.stem}", nodes, raw_edges)
        if not gt.nodes:
            skipped.append((graphml_path.stem, "no symbol nodes"))
            continue

        img = Image.open(png_path).convert("L")
        page_gray = np.array(img)

        ids = list(gt.nodes.keys())
        bboxes = [tuple(int(v) for v in gt.nodes[nid].bbox) for nid in ids]

        page_graph = process_page(page_index=0, page_gray=page_gray,
                                  symbol_bboxes=bboxes, symbol_det_ids=ids)
        j2d = build_junction_to_detection_map(page_graph.junctions, bboxes, ids)

        before = build_topology_relations(page_graph, ids, junction_to_detection_id=j2d)
        before_score = score_topology(gt, {r.pair for r in before}, sheet_group=group)
        before_results.append(before_score)

        symbol_class_of = {nid: gt.nodes[nid].cls for nid in ids}
        passthrough = {"valve", "instrumentation", "pump"}
        after = build_topology_relations(
            page_graph, ids, junction_to_detection_id=j2d,
            symbol_class_of=symbol_class_of, passthrough_symbol_classes=passthrough)
        after_score = score_topology(gt, {r.pair for r in after}, sheet_group=group)
        after_results.append(after_score)

        elapsed = time.monotonic() - t_start
        print(f"[{i+1}/{len(sheets)}] {group}/{graphml_path.stem}: "
             f"{len(gt.nodes)} nodes, {len(gt.edge_pairs)} gt edges -- "
             f"F1 {before_score.f1:.3f} -> {after_score.f1:.3f}  ({elapsed:.0f}s elapsed)")
    except Exception as e:
        skipped.append((graphml_path.stem, f"{type(e).__name__}: {e}"))
        print(f"[{i+1}/{len(sheets)}] {group}/{graphml_path.stem}: SKIPPED ({type(e).__name__}: {e})")

print(f"\n{len(skipped)} sheets skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

print("\n" + "=" * 70)
print("AGGREGATE -- BEFORE (no backbone pass)")
print("=" * 70)
agg_before = aggregate(before_results)
print(format_report("R1+R2a", agg_before))

print("\n" + "=" * 70)
print("AGGREGATE -- AFTER (backbone pass enabled)")
print("=" * 70)
agg_after = aggregate(after_results)
print(format_report("R1+R2a+backbone", agg_after))

print(f"\n\nDELTA: F1 {agg_before.f1:.3f} -> {agg_after.f1:.3f}  "
     f"(tp {agg_before.tp}->{agg_after.tp}, fp {agg_before.fp}->{agg_after.fp}, "
     f"fn {agg_before.fn}->{agg_after.fn})")
print(f"sheets scored: {len(before_results)}  total elapsed: {time.monotonic()-t_start:.0f}s")
