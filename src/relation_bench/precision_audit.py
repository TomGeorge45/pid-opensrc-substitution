"""Precision adjudication — measure what the pipeline INVENTS, not just what it finds.

Why this module exists. Every recall number this project has on a real sheet came from a
hand-trace that asked one question: of the real connections, how many did we find? On
PX-2368-0180004-001 that gave 15/26 strict (58%) and 20/26 crediting near-misses (77%). It
never asked the other half — of everything the pipeline CLAIMED, how much was real. So
precision on real sheets is, to date, entirely unmeasured.

That gap is not academic. `CLAUDE.md` rule 6 exists precisely to forbid reading recall as a
score: *"never recall alone. A model that emits thousands of boxes scores near-perfect recall
with garbage precision."* And where precision HAS been measured, it is poor:
    OPEN100/0 self-test          P=0.375  (tp=33, fp=55) — roughly 6 in 10 claims wrong
    backbone-added edges, real    0-1 of 5 correct
    12-sheet OPEN100 aggregate    F1=0.225, against a 0.226 nearest-neighbour floor
That last pair is the sobering one: the full pipeline scored about the same as connecting every
symbol to whatever is nearest. Recall alone cannot detect that; precision can.

What this produces: a numbered worklist of every claimed edge with one annotated crop each, so
a human can mark real / not-real / unsure and get a precision figure. Deliberately NOT a model
judging its own output — the local relation validator is measured degenerate (kept 0/8 on all
3 sheets, both configs; 52.6% in Probe 2, exactly the FALSE base rate), and per
`Benchmark_Gaps_Register.md` gap #3+5 a party being benchmarked should not author its own
ground truth.

The ISA-series check below is an automated PRE-FILTER, not a verdict: it ranks the worklist so
the likeliest errors are adjudicated first, and it never marks anything real or false by itself.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

BBox = Tuple[float, float, float, float]

# ISA instrument/equipment tags on these sheets carry a loop/unit number that already encodes
# which process unit the device belongs to: PSV-0300A belongs with NBK-0300, LSHL-0100 with
# MBD-0100. That is a signal completely independent of geometry, and on the one adjudicated
# sheet it would have caught real errors on its own: all three mis-bound safety devices
# (PSV-0300A, FSV-0300C, PSV-0300C) are 0300-series but were bound to 0200-series pumps.
# Essentially prod's free ISA loop-grouping prior, re-aimed at endpoint validation.
_SERIES_RE = re.compile(r"(\d{3,4})")


def tag_series(text: Optional[str]) -> Optional[str]:
    """The loop/unit number in a tag, e.g. 'PSV-0300A' -> '0300'. None when absent."""
    if not text:
        return None
    m = _SERIES_RE.search(text)
    return m.group(1) if m else None


# Types that BELONG to a piece of equipment, as opposed to sitting alongside it in the process.
# The series check only applies to these, and the reason is a measured false-alarm rate:
# a first cut that flagged ANY series mismatch raised 5 flags on PX-2368-0180004-001, and at
# least two were known-REAL connections — `PBA-0201`<->`PBA-0202` (both pumps, recorded as
# FOUND CLEANLY in the hand trace) and `NBK-0300`<->`PBA-0202` (the pump-discharge-into-treater
# line, recorded as a genuine connection the tracer caught and the LLM missed). That is correct
# behaviour for real drawings: two different process units are SUPPOSED to connect to each
# other, so equipment<->equipment series mismatch carries no information at all.
# The signal only exists in one direction — an instrument/valve/safety device carries the loop
# number of the equipment it serves, so `PSV-0300A` attached to `PBA-0201` genuinely is
# suspicious. Restricting to that case keeps the one true catch and drops both false alarms.
CHILD_TYPES_FOR_SERIES = {"instrument", "valve", "safety_device"}


def series_disagrees(a_text: Optional[str], a_type: Optional[str],
                     b_text: Optional[str], b_type: Optional[str]) -> bool:
    """True when a child device's loop number contradicts the equipment it's bound to.

    Requires exactly one child-type end and one equipment end, and both series present — a
    missing series is not evidence, and must never be treated as some.
    """
    pairs = ((a_text, a_type, b_text, b_type), (b_text, b_type, a_text, a_type))
    for child_text, child_type, equip_text, equip_type in pairs:
        if child_type in CHILD_TYPES_FOR_SERIES and equip_type == "equipment":
            cs, es = tag_series(child_text), tag_series(equip_text)
            if cs and es and cs != es:
                return True
    return False


@dataclass
class AuditItem:
    """One claimed edge awaiting a human verdict."""
    index: int
    a_id: str
    b_id: str
    a_text: Optional[str]
    b_text: Optional[str]
    a_type: Optional[str]
    b_type: Optional[str]
    edge_class: str                 # on_sheet | boundary
    source: str                     # provenance from the builder (direct segment vs traversal)
    confidence: float
    a_extent_source: Optional[str] = None
    b_extent_source: Optional[str] = None
    series_disagreement: bool = False   # ISA pre-filter flag, NOT a verdict
    gap_px: Optional[float] = None
    crop_path: Optional[str] = None
    verdict: Optional[str] = None       # real | not_real | unsure — filled in by a human
    note: Optional[str] = None

    @property
    def is_traversal(self) -> bool:
        """Traversal-derived edges rest on weaker evidence than a directly traced segment.
        Worth splitting on: on the LSHL/TSH cases, v2's CORRECT vessel edges came from direct
        `line_segment_*` evidence while the surviving WRONG HAM edges came from
        `path_through_N_segments`."""
        return self.source.startswith("path_through_")


@dataclass
class AuditWorklist:
    sheet_id: str
    entity_mode: str
    config: str
    items: List[AuditItem] = field(default_factory=list)

    # ── measured once verdicts are filled in ──────────────────────────────────
    def precision(self, *, edge_class: Optional[str] = None,
                  count_unsure_as: Optional[str] = None) -> Optional[float]:
        """Precision over adjudicated items. Returns None when nothing is adjudicated yet.

        `count_unsure_as`: None (default) excludes 'unsure' from the denominator entirely;
        'real'/'not_real' forces them, so the two bounds can be reported as a range rather
        than a single number that quietly buries the ambiguous cases.
        """
        pool = [i for i in self.items
                if i.verdict and (edge_class is None or i.edge_class == edge_class)]
        if not pool:
            return None
        real = notreal = 0
        for i in pool:
            v = i.verdict
            if v == "unsure":
                if count_unsure_as is None:
                    continue
                v = count_unsure_as
            if v == "real":
                real += 1
            elif v == "not_real":
                notreal += 1
        total = real + notreal
        return (real / total) if total else None

    def summary(self) -> Dict[str, object]:
        adjudicated = [i for i in self.items if i.verdict]
        lo = self.precision(count_unsure_as="not_real")
        hi = self.precision(count_unsure_as="real")
        return {
            "sheet_id": self.sheet_id,
            "entity_mode": self.entity_mode,
            "config": self.config,
            "claims_total": len(self.items),
            "adjudicated": len(adjudicated),
            "unsure": sum(1 for i in adjudicated if i.verdict == "unsure"),
            "precision_excl_unsure": self.precision(),
            "precision_range": None if lo is None else (round(lo, 4), round(hi, 4)),
            "precision_on_sheet": self.precision(edge_class="on_sheet"),
            "precision_boundary": self.precision(edge_class="boundary"),
            "precision_direct": self._precision_where(lambda i: not i.is_traversal),
            "precision_traversal": self._precision_where(lambda i: i.is_traversal),
            "series_disagreement_flagged": sum(1 for i in self.items if i.series_disagreement),
        }

    def _precision_where(self, pred) -> Optional[float]:
        pool = [i for i in self.items if i.verdict in ("real", "not_real") and pred(i)]
        if not pool:
            return None
        return sum(1 for i in pool if i.verdict == "real") / len(pool)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"summary": self.summary(),
                       "items": [asdict(i) for i in self.items]}, fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "AuditWorklist":
        with open(path) as fh:
            blob = json.load(fh)
        s = blob["summary"]
        wl = cls(sheet_id=s["sheet_id"], entity_mode=s["entity_mode"], config=s["config"])
        wl.items = [AuditItem(**it) for it in blob["items"]]
        return wl


def _bbox_gap(a: Optional[BBox], b: Optional[BBox]) -> Optional[float]:
    """Shortest gap between two boxes (0 when they overlap). A large gap on a claimed direct
    connection is worth a closer look."""
    if a is None or b is None:
        return None
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0.0, max(by0 - ay1, ay0 - by1))
    return round((dx * dx + dy * dy) ** 0.5, 1)


def build_worklist(
    result,
    entity_set,
    *,
    sheet_id: str,
    config: str = "v2",
) -> AuditWorklist:
    """Turn a `relationship_pipeline.V2Result` into an adjudication worklist.

    Ordering is the point: items the ISA-series check disagrees with come first, then
    traversal-derived edges (weaker evidence), then the rest by descending endpoint gap. So a
    partial adjudication is still informative — if you only get through the first 20, you have
    looked at the 20 most likely to be wrong, and `summary()` reports how many of the total
    were actually adjudicated rather than implying full coverage.
    """
    symbols = entity_set.symbol_by_id()
    ports = entity_set.port_by_id()
    extent_src = entity_set.extent_source_of()

    def _text(nid: str) -> Optional[str]:
        if nid in symbols:
            return symbols[nid].tag
        if nid in ports:
            return ports[nid].ref_tag or ports[nid].ref_text
        return None

    def _type(nid: str) -> Optional[str]:
        if nid in symbols:
            return symbols[nid].type
        return "port" if nid in ports else None

    def _extent(nid: str) -> Optional[BBox]:
        if nid in symbols:
            return symbols[nid].extent
        if nid in ports:
            return ports[nid].extent
        return None

    port_ids = entity_set.terminal_ids()
    items: List[AuditItem] = []
    for rel in result.relations:
        a, b = rel.a, rel.b
        ta, tb = _text(a), _text(b)
        edge_class = "boundary" if ((a in port_ids) != (b in port_ids)) else "on_sheet"
        items.append(AuditItem(
            index=0,
            a_id=a, b_id=b, a_text=ta, b_text=tb,
            a_type=_type(a), b_type=_type(b),
            edge_class=edge_class,
            source=rel.source, confidence=rel.confidence,
            a_extent_source=extent_src.get(a), b_extent_source=extent_src.get(b),
            series_disagreement=series_disagrees(ta, _type(a), tb, _type(b)),
            gap_px=_bbox_gap(_extent(a), _extent(b)),
        ))

    items.sort(key=lambda i: (
        not i.series_disagreement,          # flagged first
        not i.is_traversal,                  # then weaker-evidence traversal edges
        -(i.gap_px or 0.0),                  # then widest endpoint gap
    ))
    for n, it in enumerate(items, start=1):
        it.index = n

    return AuditWorklist(sheet_id=sheet_id, entity_mode=result.entity_mode,
                         config=config, items=items)


def render_audit_crops(
    worklist: AuditWorklist,
    entity_set,
    pdf_path: str,
    out_dir: str,
    *,
    render_dpi: int,
    page_index: int = 0,
    limit: Optional[int] = None,
    pad_px: int = 260,
) -> int:
    """One annotated crop per claimed edge: endpoint A boxed red, endpoint B boxed blue, framed
    to include both plus context. Returns how many were written.

    Box colours and the two-endpoint framing deliberately match the Probe 2 / R4 bundle
    convention so a human adjudicating here is looking at the same presentation the validator
    was tested on. `limit` caps the render for a partial pass — the count is returned so the
    caller can state coverage rather than imply completeness.
    """
    import fitz  # local import: only needed when actually rendering

    symbols = entity_set.symbol_by_id()
    ports = entity_set.port_by_id()

    def _extent(nid: str) -> Optional[BBox]:
        if nid in symbols:
            return symbols[nid].extent
        if nid in ports:
            return ports[nid].extent
        return None

    import os
    os.makedirs(out_dir, exist_ok=True)
    zoom = render_dpi / 72.0
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        from PIL import Image, ImageDraw
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        written = 0
        todo = worklist.items[:limit] if limit else worklist.items
        for it in todo:
            ea, eb = _extent(it.a_id), _extent(it.b_id)
            if ea is None or eb is None:
                continue
            x0 = max(0, int(min(ea[0], eb[0])) - pad_px)
            y0 = max(0, int(min(ea[1], eb[1])) - pad_px)
            x1 = min(img.width, int(max(ea[2], eb[2])) + pad_px)
            y1 = min(img.height, int(max(ea[3], eb[3])) + pad_px)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img.crop((x0, y0, x1, y1)).copy()
            d = ImageDraw.Draw(crop)
            d.rectangle([ea[0] - x0, ea[1] - y0, ea[2] - x0, ea[3] - y0],
                        outline=(220, 30, 30), width=6)
            d.rectangle([eb[0] - x0, eb[1] - y0, eb[2] - x0, eb[3] - y0],
                        outline=(30, 80, 220), width=6)
            name = (f"{it.index:03d}__{it.edge_class}__{it.a_id}_{it.b_id}"
                    f"{'__SERIES_FLAG' if it.series_disagreement else ''}.png")
            path = os.path.join(out_dir, name)
            crop.save(path)
            it.crop_path = path
            written += 1
        return written
    finally:
        doc.close()


def format_worklist_header(worklist: AuditWorklist) -> str:
    s = worklist.summary()
    return (
        f"{s['sheet_id']} [{s['entity_mode']} / {s['config']}] — {s['claims_total']} claimed "
        f"edges to adjudicate ({s['series_disagreement_flagged']} ISA-series flagged, "
        f"{sum(1 for i in worklist.items if i.is_traversal)} traversal-derived).\n"
        f"Mark each item's `verdict` as real | not_real | unsure, then call summary() — "
        f"precision is reported as a RANGE over the unsure cases, never a single number that "
        f"hides them."
    )
