"""Gap #20 — uniform cost/latency accounting across all 3 arms.

Reuses REAL, already-built prod utilities rather than inventing new tracking:
  - `pnid_pipeline.llm_proxy.snapshot`/`delta`/`usage_cost` — both real `call_llm`
    builders (`llm_proxy.build_call_llm` for prod, `extraction_local.qwen_call_llm.
    build_qwen_call_llm` for local) maintain the identical `.usage` dict shape
    (`{model: {"in","cache_w","cache_r","out","calls"}}`, confirmed by direct code read),
    so `snapshot`/`delta` work unmodified on either call_llm object.
  - `usage_cost` only applies to the PROD arm — local GPU inference has no per-token $
    cost, so the local arm's `cost_usd` is always 0.0 regardless of its usage dict.

Pipeline 3 (R1/R2a/R2b) is deterministic CV/graph code with no LLM calls at all, so its
cost_usd is always 0.0 — only wall-clock latency is meaningful there, same convention as
CLAUDE.md's existing `results.csv` schema (`vram_gb,latency_s_per_tile`, no $ column for
local compute).

One record per (arm, sheet) — "recorded uniformly" per the gap's own wording — appended to
a CSV so results accumulate across runs without re-inventing Stage 4's MLflow-free
`results.csv` convention (CLAUDE.md rule 8), just scoped to this benchmark's own file.
"""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

RELATION_BENCH_RESULTS_CSV = Path(__file__).parent / "relation_bench_results.csv"

_CSV_FIELDS = ["arm", "sheet_id", "stage", "latency_s", "cost_usd", "llm_calls", "notes"]


@dataclass
class CostLatencyRecord:
    arm: str            # "pipeline1_prod" | "pipeline2_local" | "pipeline3_proposed"
    sheet_id: str
    stage: str           # e.g. "R1", "R2a", "R2b", "hierarchy_shim", "full"
    latency_s: float
    cost_usd: float = 0.0
    llm_calls: int = 0
    notes: str = ""


@contextmanager
def timer() -> Iterator[Dict[str, float]]:
    """Wall-clock timer. `out["elapsed_s"]` is populated on context exit."""
    out: Dict[str, float] = {}
    t0 = time.monotonic()
    try:
        yield out
    finally:
        out["elapsed_s"] = time.monotonic() - t0


def deterministic_cv_record(sheet_id: str, stage: str, elapsed_s: float,
                             notes: str = "") -> CostLatencyRecord:
    """Pipeline 3 (R1/R2a/R2b): no LLM calls, cost is always $0."""
    return CostLatencyRecord(arm="pipeline3_proposed", sheet_id=sheet_id, stage=stage,
                              latency_s=round(elapsed_s, 3), cost_usd=0.0, llm_calls=0,
                              notes=notes)


def llm_arm_record(
    arm: str,
    sheet_id: str,
    stage: str,
    call_llm: Any,
    usage_before: Dict[str, Dict[str, int]],
    elapsed_s: float,
    *,
    is_prod: bool,
    notes: str = "",
) -> CostLatencyRecord:
    """Pipelines 1/2 via the gap-#12 shim. Pass `snapshot(call_llm)` taken BEFORE the run
    as `usage_before`; this computes the delta and, for the prod arm only, converts it to
    a real $ cost via the same rate table prod itself uses."""
    from pnid_pipeline.llm_proxy import delta, snapshot, usage_cost

    usage_after = snapshot(call_llm)
    d = delta(usage_before, usage_after)
    n_calls = sum(u.get("calls", 0) for u in d.values())
    cost = usage_cost(d) if is_prod else 0.0
    return CostLatencyRecord(arm=arm, sheet_id=sheet_id, stage=stage,
                              latency_s=round(elapsed_s, 3), cost_usd=cost,
                              llm_calls=n_calls, notes=notes)


def append_record(record: CostLatencyRecord, csv_path: Optional[Path] = None) -> None:
    path = csv_path or RELATION_BENCH_RESULTS_CSV
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(record))
