"""
Loads the frozen POC holdout (e2e_holdout_ids.json). See E2E_Harness_Plan.md §2 for why
this is the "option (c)" seed-disjoint split, not a genuinely file-disjoint one, and what
that means specifically for relation-set F1 (not detection/entity scoring).
"""
import json
from pathlib import Path

_HOLDOUT_PATH = Path(__file__).parent / "e2e_holdout_ids.json"

# Local cache path used throughout this project's prior PID2Graph work this session.
# The harness's data-fetch step (not yet built) should pull this from HF instead of
# assuming a local scratchpad path survives across sessions - kept explicit here so
# that TODO is visible, not silently baked in.
_PID2GRAPH_COMPLETE_ROOT = (
    "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6"
    "/scratchpad/pid2graph_inspect/PID2Graph/Complete"
)


def load_holdout() -> list:
    with open(_HOLDOUT_PATH) as f:
        data = json.load(f)
    return data["sheets"]


def holdout_sheet_paths() -> list:
    """Returns [{"sheet_id": str, "graphml_path": str, "png_path": str}, ...]"""
    out = []
    for s in load_holdout():
        base = f"{_PID2GRAPH_COMPLETE_ROOT}/{s['tree']}/{s['stem']}"
        sheet_id = f"{s['tree'].replace(' ', '')}_{s['stem']}"
        out.append({
            "sheet_id": sheet_id,
            "graphml_path": f"{base}.graphml",
            "png_path": f"{base}.png",
        })
    return out
