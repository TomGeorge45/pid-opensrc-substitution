"""
Agent output (BundleEntity/BundleRelation) vs PID2Graph ground truth -> entity F1 / relation
F1 (E2E_Harness_Plan.md §4.2, H5, H7). This is the scoring direction OPPOSITE the conversion
layer (model -> agent schema); this goes agent schema -> ground truth.

Coordinate space: PID2Graph graphml bboxes and the agent's page-raster pixel space are
assumed to be the SAME (both are "pixels in the original page image", and the harness
builds its DrawingDocument.raster directly from the same PNG PID2Graph ships) - this only
holds because the harness loads the PID2Graph PNG as-is with no resize before running the
agent's pipeline on it. If a resize/normalization step is ever added, this assumption must
be re-checked (plan §7 item 2).
"""
from dataclasses import dataclass

from .ground_truth import GTEdge, GTEntity

# Agent benchmark-ontology entity_type -> PID2Graph GT label. Non-equipment GT labels
# (connector/crossing/arrow/background) have no agent-side equivalent - excluded from GT
# for this score (mirrors the skid-grouping precedent: these aren't real equipment).
AGENT_TO_GT_LABEL = {
    "valve": "valve",
    "instrumentation": "instrumentation",
    "pump": "pump",
    "tank": "tank",
    "general": "general",
    "inlet_outlet": "inlet/outlet",
    "asset": "general",  # no direct GT equivalent for "asset" (a benchmark-only umbrella
                        # type for skid membership) - loosely mapped to "general" for
                        # entity-F1 purposes only; document this as an approximation.
}


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class EntityMatchResult:
    matched_pairs: list  # list[(agent_entity, gt_entity)]
    unmatched_agent: list
    unmatched_gt: list

    @property
    def precision(self):
        n_pred = len(self.matched_pairs) + len(self.unmatched_agent)
        return len(self.matched_pairs) / n_pred if n_pred else 0.0

    @property
    def recall(self):
        n_gt = len(self.matched_pairs) + len(self.unmatched_gt)
        return len(self.matched_pairs) / n_gt if n_gt else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_entities(agent_entities: list, gt_entities: list, iou_threshold: float = 0.5) -> EntityMatchResult:
    """agent_entities: list of BundleEntity (needs .entity_type, .source_bbox).
    gt_entities: list[GTEntity]. Greedy match, highest IoU first, type must map to the
    same GT label (AGENT_TO_GT_LABEL)."""
    candidates = []
    for ai, ae in enumerate(agent_entities):
        gt_label = AGENT_TO_GT_LABEL.get(ae.entity_type)
        if gt_label is None or not ae.source_bbox:
            continue
        for gi, ge in enumerate(gt_entities):
            if ge.label != gt_label:
                continue
            iou = _iou(ae.source_bbox, ge.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, ai, gi))
    candidates.sort(reverse=True, key=lambda c: c[0])

    used_agent, used_gt = set(), set()
    matched_pairs = []
    for iou, ai, gi in candidates:
        if ai in used_agent or gi in used_gt:
            continue
        used_agent.add(ai)
        used_gt.add(gi)
        matched_pairs.append((agent_entities[ai], gt_entities[gi]))

    unmatched_agent = [e for i, e in enumerate(agent_entities) if i not in used_agent]
    unmatched_gt = [e for i, e in enumerate(gt_entities) if i not in used_gt]
    return EntityMatchResult(matched_pairs, unmatched_agent, unmatched_gt)


@dataclass
class RelationMatchResult:
    matched_pairs: list
    unmatched_agent: list
    unmatched_gt: list

    @property
    def precision(self):
        n_pred = len(self.matched_pairs) + len(self.unmatched_agent)
        return len(self.matched_pairs) / n_pred if n_pred else 0.0

    @property
    def recall(self):
        n_gt = len(self.matched_pairs) + len(self.unmatched_gt)
        return len(self.matched_pairs) / n_gt if n_gt else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_relations(agent_relations: list, gt_edges: list, entity_match: EntityMatchResult) -> RelationMatchResult:
    """agent_relations: list of BundleRelation (needs .source_temp_id, .target_temp_id).
    gt_edges: list[GTEdge]. Undirected match by node-pair membership (PID2Graph edges have
    no forward/reverse semantics - frozenset((a,b)) is the edge identity, matching the
    convention already used in this project's earlier build_relation_pool code).

    Requires entity_match (from match_entities) to translate agent temp_ids to GT node_ids
    via the already-established entity correspondence - a relation can only be scored if
    BOTH its endpoints were themselves matched to a real GT entity."""
    temp_id_to_gt_node = {}
    for agent_entity, gt_entity in entity_match.matched_pairs:
        temp_id_to_gt_node[agent_entity.temp_id] = gt_entity.node_id

    gt_edge_set = {frozenset((e.source_node_id, e.target_node_id)) for e in gt_edges}

    matched_pairs, unmatched_agent = [], []
    used_gt_edges = set()
    for rel in agent_relations:
        src_gt = temp_id_to_gt_node.get(rel.source_temp_id)
        tgt_gt = temp_id_to_gt_node.get(rel.target_temp_id)
        if src_gt is None or tgt_gt is None:
            unmatched_agent.append(rel)
            continue
        key = frozenset((src_gt, tgt_gt))
        if key in gt_edge_set and key not in used_gt_edges:
            used_gt_edges.add(key)
            matched_pairs.append((rel, key))
        else:
            unmatched_agent.append(rel)

    unmatched_gt = [e for e in gt_edges if frozenset((e.source_node_id, e.target_node_id)) not in used_gt_edges]
    return RelationMatchResult(matched_pairs, unmatched_agent, unmatched_gt)
