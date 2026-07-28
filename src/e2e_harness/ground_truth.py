"""
PID2Graph graphml -> ground truth entities/edges for scoring (E2E_Harness_Plan.md §4.1).

Unlike the earlier skid-grouping work this project did, this does NOT filter to
equipment-only node types - entity-F1 scoring needs the real full node set (including
connector/crossing/arrow/background) so precision/recall reflect the whole graph. Whether
those non-equipment node types count against precision is a real open decision (plan §7
item 3) - kept explicit here via `EQUIPMENT_LABELS` so the matcher can apply either policy
without re-parsing.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass

EQUIPMENT_LABELS = {"valve", "instrumentation", "pump", "tank", "general", "inlet/outlet"}
NON_EQUIPMENT_LABELS = {"connector", "crossing", "arrow", "background"}


@dataclass
class GTEntity:
    node_id: str
    label: str
    bbox: list  # [x0, y0, x1, y1] floats, PID2Graph page coords


@dataclass
class GTEdge:
    source_node_id: str
    target_node_id: str
    edge_label: str  # "solid" | "non-solid"


def parse_graphml_ground_truth(graphml_path: str):
    """Returns (entities: List[GTEntity], edges: List[GTEdge]) - EVERY node/edge, no
    filtering. Caller decides (via EQUIPMENT_LABELS) whether to restrict scoring to
    equipment-type nodes only."""
    root = ET.parse(graphml_path).getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    keymap = {k.get("id"): k.get("attr.name") for k in root.findall("g:key", ns)}

    entities = []
    for node in root.iter("{http://graphml.graphdrawing.org/xmlns}node"):
        vals = {keymap.get(d.get("key"), ""): d.text for d in node.findall("g:data", ns)}
        try:
            bbox = [float(vals["xmin"]), float(vals["ymin"]),
                   float(vals["xmax"]), float(vals["ymax"])]
        except (KeyError, TypeError, ValueError):
            continue
        entities.append(GTEntity(node_id=node.get("id"), label=vals.get("label", ""), bbox=bbox))

    edges = []
    for e in root.iter("{http://graphml.graphdrawing.org/xmlns}edge"):
        vals = {keymap.get(d.get("key"), ""): d.text for d in e.findall("g:data", ns)}
        edges.append(GTEdge(
            source_node_id=e.get("source"), target_node_id=e.get("target"),
            edge_label=vals.get("edge_label", ""),
        ))
    return entities, edges


def equipment_only(entities: list) -> list:
    return [e for e in entities if e.label in EQUIPMENT_LABELS]


def contract_to_equipment_edges(entities: list, edges: list) -> list:
    """Collapse PID2Graph's routing graph (equipment nodes joined by chains of
    connector/crossing nodes representing drawn pipe polylines) down to DIRECT
    equipment<->equipment edges, by walking each chain of non-equipment nodes and
    connecting the equipment nodes at its ends.

    **Load-bearing discovery (2026-07-16):** verified directly on a real sheet
    (Dataset PID/246, 445 edges) that PID2Graph contains ZERO direct
    equipment<->equipment edges - literally every edge touches at least one
    connector/crossing node. A real agent BundleRelation always connects two
    equipment entities directly (e.g. "valve X connects to pump Y") - line tracing
    collapses a drawn pipe run into one direct relation, it never emits an
    intermediate synthetic node. Scoring BundleRelations against PID2Graph's raw
    edges therefore CANNOT produce a nonzero relation-F1 by construction, on ANY
    sheet, regardless of how good detection/relation-building is - this is what
    caused relation_f1=0.0 across all 4 holdout sheets in the v3 cascade run,
    including sheets with 60%+ entity recall and 100+ relations built. This
    function must run once on the GT edges before match_relations is used.

    **Tunneling rule (degree-4 heuristic, chosen 2026-07-16 after comparing 3
    variants on real data - Dataset PID/246, 96 equipment nodes):**
      - "connector" nodes: always tunnel through (every one measured was degree-2,
        i.e. an unambiguous mid-line waypoint).
      - "crossing" nodes: tunnel through UNLESS degree==4. A degree-4 crossing is
        the classic P&ID "two independent lines visually cross, do not connect"
        symbol (4 line-ends = 2 unrelated through-paths); degree-2/3 crossings
        behave like real junctions/waypoints in this data and are tunneled through.
      Rejected alternatives, both measured directly: tunneling through EVERY
      connector/crossing node produces 521 contracted edges / mean degree ~11 on
      this sheet (unrealistic - collapses shared headers into all-pairs cliques);
      blocking at EVERY crossing node produces only 13 edges / mean degree 0.27
      (most equipment ends up isolated - real connections routinely cross other
      lines en route). The degree-4-only-blocks rule produced 92 edges / mean
      degree 1.92 / max degree 5 - the only variant with a plausible P&ID
      connectivity profile. Still a heuristic, not verified against the original
      drawing's actual crossing geometry - flag this if a sheet's relation-F1
      looks suspicious."""
    adjacency = {}
    for e in edges:
        adjacency.setdefault(e.source_node_id, []).append(e.target_node_id)
        adjacency.setdefault(e.target_node_id, []).append(e.source_node_id)

    label_by_id = {ent.node_id: ent.label for ent in entities}
    equip_ids = {nid for nid, lbl in label_by_id.items() if lbl in EQUIPMENT_LABELS}
    degree = {nid: len(neighbors) for nid, neighbors in adjacency.items()}

    def is_tunnelable(nid: str) -> bool:
        lbl = label_by_id.get(nid)
        if lbl == "connector":
            return True
        if lbl == "crossing":
            return degree.get(nid, 0) != 4
        return False

    contracted = set()
    for start in equip_ids:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adjacency.get(node, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt in equip_ids:
                    if nxt != start:
                        contracted.add(frozenset((start, nxt)))
                    # do not expand past an equipment node - it terminates this path
                elif is_tunnelable(nxt):
                    stack.append(nxt)
                # else: hard stop (e.g. degree-4 crossing) - not a real connection

    return [GTEdge(source_node_id=a, target_node_id=b, edge_label="contracted")
            for a, b in (tuple(pair) for pair in contracted)]
