"""Real before/after F1 measurement for the gap #14 backbone pass, using the SAME
OPEN100/0 sheet + methodology already self-tested for gap #10 (R1+R2a F1=0.104 baseline).
R1 (raster tracer) runs ONCE; R2a (build_topology_relations) runs twice -- with and
without the backbone-pass params -- so any F1 delta is attributable ONLY to the new pass.
"""
import sys
sys.path.insert(0, "/Users/tomgeorge/pid-ml/src/relation_bench")

from pathlib import Path
import numpy as np
from PIL import Image

from pid2graph_gt import parse_graphml, contract
from line_tracing import process_page
from graph_construction import build_topology_relations, build_junction_to_detection_map
from score import score_topology, format_report

ROOT = Path("/System/Volumes/Data/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6/scratchpad/molmo_ft_prep_v2/pid2graph/PID2Graph/Complete/PID2Graph OPEN100")
graphml_path = ROOT / "0.graphml"
png_path = ROOT / "0.png"

nodes, raw_edges, _ = parse_graphml(graphml_path)
gt = contract("OPEN100/0", nodes, raw_edges)
print(f"GT: {len(gt.nodes)} symbol nodes, {len(gt.edge_pairs)} contracted edges")

img = Image.open(png_path).convert("L")
page_gray = np.array(img)
print(f"page_gray shape: {page_gray.shape}")

ids = list(gt.nodes.keys())
bboxes = [tuple(int(v) for v in gt.nodes[nid].bbox) for nid in ids]

page_graph = process_page(
    page_index=0, page_gray=page_gray,
    symbol_bboxes=bboxes, symbol_det_ids=ids,
)
print(f"R1 output: {len(page_graph.segments)} segments, {len(page_graph.junctions)} junctions")

j2d = build_junction_to_detection_map(page_graph.junctions, bboxes, ids)
print(f"junctions resolved to symbols: {len(j2d)}")

# --- BEFORE: no backbone pass (reproduces the existing gap #10 self-test) ---
before = build_topology_relations(page_graph, ids, junction_to_detection_id=j2d)
before_pairs = {r.pair for r in before}
before_score = score_topology(gt, before_pairs)
print("\n=== BEFORE (no backbone pass) ===")
print(format_report("R1+R2a", before_score))

# --- AFTER: backbone pass enabled ---
symbol_class_of = {nid: gt.nodes[nid].cls for nid in ids}
passthrough = {"valve", "instrumentation", "pump"}
after = build_topology_relations(
    page_graph, ids, junction_to_detection_id=j2d,
    symbol_class_of=symbol_class_of, passthrough_symbol_classes=passthrough,
)
after_pairs = {r.pair for r in after}
after_score = score_topology(gt, after_pairs)
print("\n=== AFTER (backbone pass enabled) ===")
print(format_report("R1+R2a+backbone", after_score))

print(f"\ndelta: F1 {before_score.f1:.3f} -> {after_score.f1:.3f}  "
     f"(tp {before_score.tp}->{after_score.tp}, fp {before_score.fp}->{after_score.fp}, "
     f"fn {before_score.fn}->{after_score.fn})")

ga_key = "class:general|general"
b_ga = before_score.by_stratum.get(ga_key)
a_ga = after_score.by_stratum.get(ga_key)
if b_ga and a_ga:
    print(f"\ngeneral|general stratum: F1 {b_ga.f1:.3f} -> {a_ga.f1:.3f} "
         f"(tp {b_ga.tp}->{a_ga.tp}, fp {b_ga.fp}->{a_ga.fp}, fn {b_ga.fn}->{a_ga.fn})")
