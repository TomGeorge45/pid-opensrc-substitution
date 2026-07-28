"""Reproduce the final 2026-07-28 Pipeline 3 v2 numbers on PX-2368-0180004-001.

Exists because the reported figures depended on inputs that lived only in a chat transcript —
three hand-read equipment extents and four hand-recovered instrument seeds. Both are now in
`src/relation_bench/hand_extents/px2368.json`, and this script wires them up so the result is
regenerable rather than merely asserted.

Expected output (as reported): precision 93.9%, recall 86.7%, F1 0.902, 42 claims, 0 violations.

Needs: the sheet PDF (HF `sheets/RIVE_LTTS_Sample.zip`), the extraction JSON
(HF `benchmarks/extraction_2026-07-24/...`), and `.env` with HF_TOKEN.
Run with the venv on the path:  PYTHONPATH=src/relation_bench .venv-e2e/bin/python scripts/reproduce_px2368_final.py <pdf_path>
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

import relationship_pipeline as RP
from entities import SymbolNode, classify_prod_tags, detect_ports_from_sheet_refs
from extent_resolution import (extract_page_vector_paths, resolve_extent_from_seed,
                               resolve_extents, seed_from_bbox_center)

SHEET = "PX-2368-0180004-001"
EXTRACTION = f"benchmarks/extraction_2026-07-24/{SHEET}_gpt55low.json"
HAND = "src/relation_bench/hand_extents/px2368.json"
ADJUDICATION = "benchmarks/px2368_adjudication_42claims_2026-07-27.json"


def main(pdf_path: str) -> int:
    load_dotenv()
    cfg = json.load(open(HAND))
    tok = os.environ.get("HF_TOKEN")
    d = json.load(open(hf_hub_download("timthy45/pnid-extraction-datasets", EXTRACTION,
                                       repo_type="dataset", token=tok)))
    dpi = d["drawing"]["render_dpi"]
    paths, page = extract_page_vector_paths(pdf_path, 0, render_scale=dpi / 72.0)

    es = classify_prod_tags(d["tags"], page_size=page)

    # the WIDE variant is the one that produced the reported numbers; tight lost 2 real edges
    wide = {k: v["bbox"] for k, v in cfg["equipment_extents_wide"].items()}
    for s in es.symbols:
        if s.id in wide:
            s.extent = tuple(wide[s.id])
            s.extent_source = "hand"
            s.point = ((s.extent[0] + s.extent[2]) / 2, (s.extent[1] + s.extent[3]) / 2)

    seed_from_bbox_center(es)
    resolve_extents(es, pdf_path, render_dpi=dpi)

    for i, rec in enumerate(cfg["recovered_instrument_seeds"]):
        seed = tuple(rec["seed"])
        ext, src, conf = resolve_extent_from_seed(seed, paths, render_dpi=dpi, page_size=page)
        es.symbols.append(SymbolNode(id=f"fix{i:02d}", point=seed, extent=ext,
                                     extent_source=src, extent_conf=conf, tag=rec["tag"],
                                     tag_source="hand", type=rec["type"]))

    es.ports, _ = detect_ports_from_sheet_refs(pdf_path, d["tags"], render_dpi=dpi, page_size=page)
    es.mode = "hand_verified"

    r = RP.run_relationship_pipeline_v2(es, pdf_path, dpi)

    name = {s.id: (s.tag or s.id) for s in es.symbols}
    for p in es.ports:
        name[p.id] = p.ref_tags[0] if p.ref_tags else f"sheet {p.ref_sheet}"
    have = {frozenset((x.a, x.b)) for x in r.relations}

    def got(a: str, b: str) -> bool:
        A = [k for k, v in name.items() if v == a]
        B = [k for k, v in name.items() if v == b]
        return any(frozenset((x, y)) in have for x in A for y in B)

    gt = [tuple(e) for e in cfg["independent_gt_edges"]]
    rec_hits = sum(1 for e in gt if got(*e))

    prior = json.load(open(ADJUDICATION))
    verdict = {frozenset((str(i["a_text"]), str(i["b_text"]))): i["verdict"] for i in prior["items"]}
    real = sum(1 for x in r.relations
               if verdict.get(frozenset((name.get(x.a, x.a), name.get(x.b, x.b)))) == "real")
    false = sum(1 for x in r.relations
                if verdict.get(frozenset((name.get(x.a, x.a), name.get(x.b, x.b)))) == "not_real")

    P = real / (real + false) if (real + false) else 0.0
    R = rec_hits / len(gt)
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    print(f"{SHEET} — Pipeline 3 v2 final")
    print(f"  symbols={len(es.symbols)} ports={len(es.ports)} labels={len(es.labels)}")
    print(f"  claims={len(r.relations)} (on_sheet={len(r.on_sheet)} boundary={len(r.boundary)})")
    print(f"  violations={len(r.violations)}")
    print(f"  precision={P:.1%}  recall={R:.1%} ({rec_hits}/{len(gt)})  F1={F1:.3f}")
    print(f"  expected:  precision=93.9%  recall=86.7%  F1=0.902  claims=42  violations=0")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
