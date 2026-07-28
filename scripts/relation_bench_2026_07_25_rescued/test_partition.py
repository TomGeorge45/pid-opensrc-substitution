import sys
sys.path.insert(0, "/Users/tomgeorge/pid-ml/src/relation_bench")
from agreement_diff import _llm_claimed_pairs, OffPageClaim

tags_by_id = {
    "t1": {"id": "t1", "type": "equipment"},
    "t2": {"id": "t2", "type": "valve"},
    "t3": {"id": "t3", "type": "equipment"},
}

relationships = [
    {"kind": "feeds", "from": "t1", "to": "t2"},                       # on-sheet<->on-sheet
    {"kind": "feeds", "from": "t3", "to": "MBD-0635"},                 # off-page (text fallback)
    {"kind": "relieves_to", "from": "GHOST-999", "to": "ALSO-GHOST"},  # neither resolves -- dropped
    {"kind": "hosted_by", "from": "t1", "to": "t3"},                   # wrong kind -- dropped
    {"kind": "actuates", "from": "t2", "to": "t1"},                    # dup pair, same as first (undirected)
]

pairs, off_page = _llm_claimed_pairs(tags_by_id, relationships)
print("on-sheet pairs:", pairs)
print("off-page claims:", off_page)

assert pairs == {frozenset(("t1", "t2"))}, f"FAIL: unexpected on-sheet pairs {pairs}"
assert len(off_page) == 1, f"FAIL: expected 1 off-page claim, got {len(off_page)}"
assert off_page[0] == OffPageClaim(on_sheet_id="t3", off_page_text="MBD-0635", kind="feeds")
print("\nALL CHECKS PASSED")
