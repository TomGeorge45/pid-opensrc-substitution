"""Run the updated agreement_diff.py (claim partitioning + line-label filtering, Tier 1
items 1+4) against the 3 fresh real extraction results (2026-07-24 rerun)."""
import json
import sys

sys.path.insert(0, "/Users/tomgeorge/pid-ml/src/relation_bench")
from agreement_diff import compute_agreement, format_agreement

SCRATCH = "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/10ffbddb-0f7d-4f19-a544-f1152513500c/scratchpad"
PRED_DIR = f"{SCRATCH}/preds_gpt55_partB_2026-07-24"

SHEETS = [
    ("PX-2368-0180004-001", f"{PRED_DIR}/PX-2368-0180004-001_gpt55low.json",
     f"{SCRATCH}/sheets/RIVE/PX-2368-0180004-001.pdf"),
    ("GD-B-540-DP-2920-005-Z", f"{PRED_DIR}/GD-B-540-DP-2920-005-Z_gpt55low.json",
     f"{SCRATCH}/sheets/AG_PNID/GD-B-540-DP-2920-005-Z.pdf"),
    ("PX-2365-0140006-001", f"{PRED_DIR}/PX-2365-0140006-001_gpt55low.json",
     f"{SCRATCH}/sheets/RIVE/PX-2365-0140006-001.PDF"),
]

results = []
for sheet_id, json_path, pdf_path in SHEETS:
    with open(json_path) as f:
        extraction_result = json.load(f)
    try:
        r = compute_agreement(sheet_id, extraction_result, pdf_path)
        results.append(r)
        print(format_agreement(r))
        # tag type breakdown, to see how many 'line' type tags got filtered
        types = {}
        for t in extraction_result["tags"]:
            types[t.get("type", "?")] = types.get(t.get("type", "?"), 0) + 1
        print(f"  tag types: {types}")
        print()
    except Exception as e:
        print(f"ERROR on {sheet_id}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print()

# aggregate off-page visibility across all 3 sheets
total_on_sheet = sum(r.on_sheet_claim_count for r in results)
total_off_page = sum(len(r.off_page_claims) for r in results)
total_claims = total_on_sheet + total_off_page
print("=== AGGREGATE ACROSS ALL 3 SHEETS ===")
print(f"total connectivity claims: {total_claims}")
print(f"  on-sheet<->on-sheet: {total_on_sheet} ({total_on_sheet/total_claims:.1%})" if total_claims else "n/a")
print(f"  off-page: {total_off_page} ({total_off_page/total_claims:.1%})" if total_claims else "n/a")
