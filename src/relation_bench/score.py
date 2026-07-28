"""Relation-stage scoring harness — shared by all 3 benchmark arms.

Scores a predicted edge set against a contracted PID2Graph SheetGT (see pid2graph_gt.py).
Frozen protocol decisions (Benchmark_Gaps_Register.md gaps #16, #17, #19, #21):
  - undirected pair matching (gap #16)
  - stratified by line_type (solid/dashed) and by endpoint-class pair (gap #19)
  - a trivial nearest-neighbor-connect floor baseline is provided so every real arm's
    number has a sanity floor to sit above (gap #21)
No entity detection happens here — this harness assumes entities are handed in with their
GT node ids already resolved (the whole point of a relation-only benchmark: isolate the
relation stage from the entity stage). See runner.py for how each arm gets its input.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

from pid2graph_gt import SheetGT


@dataclass
class PredEdge:
    a: str          # GT node id (ground truth entities are given, not detected here)
    b: str
    kind: str | None = None       # e.g. "feeds" / "on_line" / None if arm doesn't emit kinds


@dataclass
class ScoreResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    by_stratum: Dict[str, "ScoreResult"] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_topology(gt: SheetGT, pred_pairs: Set[FrozenSet[str]],
                    sheet_group: str | None = None) -> ScoreResult:
    """Topology-only P/R/F1 (does an edge exist between A and B), undirected, plus
    stratified breakdowns by line_type, endpoint-class pair, and (gap #19) sheet density
    group. Predicted pairs referencing a node not in gt.nodes are dropped as unscoreable
    (that node was never a real symbol per GT) rather than counted as a false positive
    against nothing.

    `sheet_group` (gap #19): an optional density-bucket label for THIS sheet — e.g.
    "sparse" (OPEN100) vs "dense" (Dataset PID), the exact split already frozen in
    Benchmark_Gaps_Register.md gap #17. Unlike line_type/class (edge-level properties, a
    pair can be any class pair), density is a SHEET-level property — every pair scored here
    belongs to the one sheet this call is scoring, so the whole result's tp/fp/fn goes into
    one `sheet_group:<label>` bucket. `aggregate()` needs no changes to support this: it
    already sums by_stratum keys generically across sheets, so density numbers accumulate
    correctly as more per-sheet ScoreResults get aggregated — a flat "R2a: F1=0.10" hides
    exactly the failure mode this benchmark cares about (does the dense/well-connected tree
    perform differently from the sparse one?), which is the whole point of gap #19."""
    gt_pairs = gt.edge_pairs
    clean_pred = {p for p in pred_pairs if len(p) == 2 and all(n in gt.nodes for n in p)}

    tp_pairs = clean_pred & gt_pairs
    fp_pairs = clean_pred - gt_pairs
    fn_pairs = gt_pairs - clean_pred

    result = ScoreResult(tp=len(tp_pairs), fp=len(fp_pairs), fn=len(fn_pairs))

    # stratify by line_type (only meaningful for tp/fn, since fp has no GT line_type —
    # bucket fp under its own "false_positive" pseudo-stratum key for that axis)
    by_line: Dict[str, ScoreResult] = {}
    for p in tp_pairs:
        lt = gt.edges[p]
        by_line.setdefault(lt, ScoreResult()).tp += 1
    for p in fn_pairs:
        lt = gt.edges[p]
        by_line.setdefault(lt, ScoreResult()).fn += 1
    by_line.setdefault("false_positive", ScoreResult()).fp = len(fp_pairs)

    # stratify by endpoint-class pair (e.g. "general|valve") — the asset<->asset money
    # metric is stratum == "general|general"
    by_class: Dict[str, ScoreResult] = {}
    for p in tp_pairs:
        by_class.setdefault(gt.stratum(p), ScoreResult()).tp += 1
    for p in fn_pairs:
        by_class.setdefault(gt.stratum(p), ScoreResult()).fn += 1
    for p in fp_pairs:
        classes = sorted(gt.nodes[n].cls for n in p)
        by_class.setdefault("|".join(classes), ScoreResult()).fp += 1

    result.by_stratum = {**{f"line:{k}": v for k, v in by_line.items()},
                         **{f"class:{k}": v for k, v in by_class.items()}}
    if sheet_group is not None:
        result.by_stratum[f"sheet_group:{sheet_group}"] = ScoreResult(
            tp=result.tp, fp=result.fp, fn=result.fn)
    return result


def nearest_neighbor_floor(gt: SheetGT, k: int = 1) -> Set[FrozenSet[str]]:
    """Trivial baseline (gap #21): connect every symbol to its k nearest OTHER symbols by
    bbox-center Euclidean distance. Not model-informed at all — any real arm scoring below
    this on recall has learned nothing beyond "things near each other are related", and
    any arm scoring below it on precision is worse than random proximity guessing."""
    ids = list(gt.nodes.keys())
    pairs: Set[FrozenSet[str]] = set()
    for nid in ids:
        cx, cy = gt.nodes[nid].center
        dists = []
        for other in ids:
            if other == nid:
                continue
            ox, oy = gt.nodes[other].center
            d = (cx - ox) ** 2 + (cy - oy) ** 2
            dists.append((d, other))
        dists.sort()
        for _d, other in dists[:k]:
            pairs.add(frozenset((nid, other)))
    return pairs


def aggregate(results: List[ScoreResult]) -> ScoreResult:
    """Micro-average across sheets (sum tp/fp/fn, not mean of per-sheet F1 — mean-of-F1
    lets a tiny sheet with 2 GT edges swing the aggregate as much as a 500-edge sheet)."""
    agg = ScoreResult()
    strata: Dict[str, ScoreResult] = {}
    for r in results:
        agg.tp += r.tp
        agg.fp += r.fp
        agg.fn += r.fn
        for k, v in r.by_stratum.items():
            s = strata.setdefault(k, ScoreResult())
            s.tp += v.tp
            s.fp += v.fp
            s.fn += v.fn
    agg.by_stratum = strata
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 3 v2 — input-mode tagging and the on-sheet / boundary split
# ══════════════════════════════════════════════════════════════════════════════

# Where the entities a score was computed over came from. Carried on every row and NEVER
# mixed into one number, because each mode answers a different question:
#   gt_injected   — perfect entities (PID2Graph GT). Isolates relation quality; the mode every
#                   existing recorded number was measured in, so the only one comparable to
#                   them. Free, and real GT.
#   hand_verified — human-checked extents on real sheets, where no entity GT exists at all.
#   detected      — Molmo2 / multi-arm union. The honest end-to-end number, and the only one
#                   that includes extraction error.
# Mixing them silently would reproduce the exact failure this project has already been burned
# by twice: a number that looks comparable to a previous number while measuring something else.
ENTITY_MODES = ("gt_injected", "hand_verified", "detected")


@dataclass
class V2ScoreRow:
    """One scored (sheet, config) result, with everything needed to read it honestly.

    `on_sheet` and `boundary` are deliberately separate ScoreResults and there is no combined
    field, because the two have different evidence standards: an on-sheet edge is confirmed by
    geometry alone, a boundary edge needs geometry PLUS a text read of the doorway (Probe 3's
    task, 87.5%). Averaging them would hide which half is carrying the number — the same
    reason CLAUDE.md rule 5 forbids averaging detection and typing.
    """
    sheet_id: str
    entity_mode: str
    config: str                                  # e.g. "v2" / "v2_no_backbone" / "v1_original"
    on_sheet: ScoreResult = field(default_factory=ScoreResult)
    boundary: ScoreResult = field(default_factory=ScoreResult)
    n_symbols: int = 0
    n_ports: int = 0
    n_labels: int = 0
    extent_source_counts: Dict[str, int] = field(default_factory=dict)
    suppressed_port_pairs: int = 0
    violations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.entity_mode not in ENTITY_MODES:
            raise ValueError(
                f"entity_mode must be one of {ENTITY_MODES}, got {self.entity_mode!r} — "
                "an untagged mode is how incomparable numbers get compared")


def score_v2(
    gt: SheetGT,
    on_sheet_pairs: Set[FrozenSet[str]],
    boundary_pairs: Set[FrozenSet[str]],
    *,
    sheet_id: str,
    entity_mode: str,
    config: str,
    extent_source_of: Dict[str, str] | None = None,
    sheet_group: str | None = None,
    n_symbols: int = 0,
    n_ports: int = 0,
    n_labels: int = 0,
    suppressed_port_pairs: int = 0,
    violations: List[str] | None = None,
) -> V2ScoreRow:
    """Score a v2 run, keeping on-sheet and boundary edges apart.

    Boundary edges are scored against `gt` only when GT actually contains the port node — which
    on PID2Graph it never does (those sheets have no border connector columns at all, the very
    reason the 762-sheet corpus check missed the border-column failure). So on PID2Graph the
    boundary result will legitimately be all-zero; that is a true statement about the corpus,
    not a bug, and it is exactly why it must not be folded into the on-sheet number.

    `extent_source_of` adds an `extent:<source>` stratum so a run can be split by extent
    quality — answering "is this number limited by R0?" instead of guessing.
    """
    on = score_topology(gt, on_sheet_pairs, sheet_group=sheet_group)
    bd = score_topology(gt, boundary_pairs, sheet_group=sheet_group)

    if extent_source_of:
        for pair in on_sheet_pairs:
            if len(pair) != 2:
                continue
            srcs = sorted({extent_source_of.get(n, "unknown") for n in pair})
            key = f"extent:{'|'.join(srcs)}"
            bucket = on.by_stratum.setdefault(key, ScoreResult())
            if pair in gt.edge_pairs:
                bucket.tp += 1
            elif all(n in gt.nodes for n in pair):
                bucket.fp += 1

    counts: Dict[str, int] = {}
    for src in (extent_source_of or {}).values():
        counts[src] = counts.get(src, 0) + 1

    return V2ScoreRow(
        sheet_id=sheet_id, entity_mode=entity_mode, config=config,
        on_sheet=on, boundary=bd,
        n_symbols=n_symbols, n_ports=n_ports, n_labels=n_labels,
        extent_source_counts=counts,
        suppressed_port_pairs=suppressed_port_pairs,
        violations=list(violations or []),
    )


def format_v2_report(row: V2ScoreRow) -> str:
    lines = [
        f"{row.sheet_id} [{row.entity_mode} / {row.config}]  "
        f"symbols={row.n_symbols} ports={row.n_ports} labels={row.n_labels}",
        f"  ON-SHEET  P={row.on_sheet.precision:.3f} R={row.on_sheet.recall:.3f} "
        f"F1={row.on_sheet.f1:.3f} (tp={row.on_sheet.tp} fp={row.on_sheet.fp} fn={row.on_sheet.fn})",
        f"  BOUNDARY  P={row.boundary.precision:.3f} R={row.boundary.recall:.3f} "
        f"F1={row.boundary.f1:.3f} (tp={row.boundary.tp} fp={row.boundary.fp} fn={row.boundary.fn})"
        "   <- geometry + doorway text read; never averaged with on-sheet",
    ]
    if row.extent_source_counts:
        lines.append("  extents: " + ", ".join(
            f"{k}={v}" for k, v in sorted(row.extent_source_counts.items())))
    if row.suppressed_port_pairs:
        lines.append(f"  suppressed port<->port pairs: {row.suppressed_port_pairs}")
    if row.violations:
        lines.append(f"  !! violations: {len(row.violations)} (first: {row.violations[0]})")
    return "\n".join(lines)


def format_report(label: str, result: ScoreResult) -> str:
    lines = [f"{label}: P={result.precision:.3f} R={result.recall:.3f} F1={result.f1:.3f} "
            f"(tp={result.tp} fp={result.fp} fn={result.fn})"]
    for k in sorted(result.by_stratum):
        v = result.by_stratum[k]
        lines.append(f"    {k:24} P={v.precision:.3f} R={v.recall:.3f} F1={v.f1:.3f} "
                    f"(tp={v.tp} fp={v.fp} fn={v.fn})")
    return "\n".join(lines)
