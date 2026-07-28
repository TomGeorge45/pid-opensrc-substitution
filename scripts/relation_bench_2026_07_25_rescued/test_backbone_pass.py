"""Self-test for gap #14 (process-backbone pass) — synthetic graph, no PDF/GPU needed.

Topology: VESSEL_A --seg1-- VALVE_1 --seg2-- VALVE_2 --seg3-- VESSEL_B
No direct short segment between VESSEL_A and VESSEL_B — the only way to discover that
relation is by walking THROUGH the two inline valves. This is exactly the asset<->asset
backbone pattern (F1 0.126 vs 0.314 valve|valve) documented in Benchmark_Gaps_Register.md.

Checks:
  1. WITHOUT the backbone params (old behavior) -- VESSEL_A<->VESSEL_B is NOT found.
  2. WITH the backbone params -- VESSEL_A<->VESSEL_B IS found (the fix).
  3. The existing short-hop pairs (VESSEL_A<->VALVE_1, VALVE_1<->VALVE_2,
     VALVE_2<->VESSEL_B) are STILL found in both cases -- no regression.
"""
import sys
sys.path.insert(0, "/Users/tomgeorge/pid-ml/src/relation_bench")
sys.path.insert(0, "/Users/tomgeorge/pid-ml/src/relation_bench/line_tracing")

from line_tracing.models import Endpoint, PageLineGraph, Segment
from graph_construction import build_topology_relations


def seg(sid, a_kind, a_ref, b_kind, b_ref):
    return Segment(
        segment_id=sid, page_index=0,
        polyline=[(0, 0), (10, 10)],
        stroke_style="continuous", line_type="process",
        endpoint_a=Endpoint(kind=a_kind, ref=a_ref, position=(0, 0)),
        endpoint_b=Endpoint(kind=b_kind, ref=b_ref, position=(10, 10)),
        confidence=0.9,
    )


graph = PageLineGraph(
    page_index=0,
    segments=[
        seg("p0_g0000", "symbol", "VESSEL_A", "symbol", "VALVE_1"),
        seg("p0_g0001", "symbol", "VALVE_1", "symbol", "VALVE_2"),
        seg("p0_g0002", "symbol", "VALVE_2", "symbol", "VESSEL_B"),
    ],
    junctions=[],
)

valid_ids = ["VESSEL_A", "VALVE_1", "VALVE_2", "VESSEL_B"]
symbol_class_of = {"VESSEL_A": "general", "VALVE_1": "valve", "VALVE_2": "valve", "VESSEL_B": "general"}
passthrough = {"valve", "instrumentation", "pump"}

target_pair = frozenset(("VESSEL_A", "VESSEL_B"))
short_pairs = [
    frozenset(("VESSEL_A", "VALVE_1")),
    frozenset(("VALVE_1", "VALVE_2")),
    frozenset(("VALVE_2", "VESSEL_B")),
]

# 1. Old behavior (no backbone params)
old = build_topology_relations(graph, valid_ids, max_path_depth=8)
old_pairs = {r.pair for r in old}
print("OLD (no backbone) pairs found:", old_pairs)
assert target_pair not in old_pairs, "FAIL: old behavior should NOT find the backbone pair"
for p in short_pairs:
    assert p in old_pairs, f"FAIL: old behavior should still find short pair {p}"
print("  -> confirmed: backbone pair absent, short pairs present (as expected, old behavior)\n")

# 2. New behavior (backbone pass enabled)
new = build_topology_relations(
    graph, valid_ids, max_path_depth=8,
    symbol_class_of=symbol_class_of, passthrough_symbol_classes=passthrough,
)
new_pairs = {r.pair for r in new}
print("NEW (with backbone) pairs found:", new_pairs)
assert target_pair in new_pairs, "FAIL: backbone pass should find VESSEL_A<->VESSEL_B"
for p in short_pairs:
    assert p in new_pairs, f"FAIL: backbone pass regressed short pair {p}"
print("  -> confirmed: backbone pair NOW found, short pairs still present (no regression)\n")

print("ALL CHECKS PASSED")
