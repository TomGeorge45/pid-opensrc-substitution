"""Pipeline 3 v2 — the entity contract Pipeline 3 consumes.

Replaces "a flat list of all tags" with three genuinely different things, because the old
flat list conflated them and that conflation was the project's most expensive bug source:

  - :class:`SymbolNode`  — a thing that is DRAWN on this sheet. Has a real extent (the shape,
                           not the name plate) and optionally a tag. A pipe can end at it,
                           and (if it's an inline fitting) a walk may pass through it.
  - :class:`PortNode`    — an off-page connector GRAPHIC. Really drawn here, at the border, and
                           a real place for a pipe to terminate — but the equipment it NAMES
                           lives on another sheet and has no location here. TERMINAL: a pipe
                           ends at a port, a walk never passes through one.
  - :class:`LabelAnnotation` — text that annotates something rather than being something. Pipe
                           specs (`6"(300#)`), notes, title-block fields. NEVER a node, NEVER
                           a tracer endpoint.

Why the port/remote split matters (Benchmark_Gaps_Register.md gap #22 + Part D): the old model
gave off-page equipment a narrow label-shaped bbox and treated it as ordinary on-sheet
equipment. Two consequences, both measured on PX-2368-0180004-001:
  1. 4 of the 5 edges the backbone pass added were pairs of *vertically adjacent entries in
     the same border column* — two unrelated doorways, joined because the walk treated them as
     inline devices. (`MBF-0623`/`HBG-0905` are 228px apart at identical x=5290;
     `MBF-0500`/`PBM-0450` 138px; `PBA-0501`/`PBA-0903` 406px at identical x≈838.)
     Making ports terminal removes this failure by construction, not by tuning.
  2. ~19-20 of ~25 true connectivity claims per sheet were declared unverifiable, because
     "does a line reach MBD-0635" is unanswerable when MBD-0635 isn't drawn here. Split into
     port + remote reference, it becomes two answerable questions: does a line reach this
     doorway (geometry), and does the doorway's text read MBD-0635 (Probe 3's task, 87.5%).

Untagged symbols are KEPT (`tag is None`). Two reasons, both real: GD-B-540-DP-2920-005-Z has
exactly ONE extractable embedded word on the whole page (its CAD export outlined text into
vector paths), so tag association will fail outright on some sheets; and both scoring corpora
(Gupta, PID2Graph) are class-agnostic anyway, so a nameless node is the native case, not an
exception. A missing name must never zero out a correct topology finding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]

# How each symbol's extent was obtained. Recorded per node so every downstream number can be
# stratified by extent quality — the whole point of R0 is that a bad extent silently poisons
# everything after it, so it must never be an unmeasured black box in the middle.
EXTENT_SOURCES = ("vector_seeded", "raster_seeded", "radius_fallback", "hand", "given_bbox")

# Tag types that annotate rather than exist. On PX-2368 this is 45 of 104 tags — 43% of what
# the old pipeline treated as connectable entities were pipe-spec labels. Removing them here
# means the old Tier-1 #4 line-filter becomes a defensive assert rather than a load-bearing
# stage (see relationship_pipeline.assert_no_label_endpoints).
LABEL_TAG_TYPES = {"line"}


@dataclass
class SymbolNode:
    """A drawn symbol. `extent` is the SHAPE's bbox, not the name plate's."""
    id: str
    point: Optional[Point] = None          # the seed (Molmo2's point), if we had one
    extent: Optional[BBox] = None          # resolved drawn-shape bbox
    extent_source: str = "given_bbox"
    extent_conf: Optional[float] = None
    tag: Optional[str] = None              # None => untagged symbol, still a full node
    tag_source: Optional[str] = None       # nearest_ocr | vlm_read | hand | prod_tag
    tag_dist_px: Optional[float] = None    # distance seed->chosen word, for auditing mispairs
    type: Optional[str] = None
    source_arm: str = "unknown"

    @property
    def is_tagged(self) -> bool:
        return bool(self.tag)

    def tracer_bbox(self) -> Optional[BBox]:
        return self.extent

    def center(self) -> Optional[Point]:
        if self.extent is None:
            return self.point
        x0, y0, x1, y1 = self.extent
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass
class PortNode:
    """An off-page connector — a doorway in the edge of the sheet.

    TERMINAL by construction. `ref_*` describe what is on the far side; they are attributes of
    the doorway, never a node with a location on this sheet.

    LOCATION IS STILL UNSOLVED. Two hypotheses tested on PX-2368-0180004-001, both refuted:

      1. *The connector graphic encloses its text, so seeded extent resolution finds it.*
         REFUTED — 11 of 14 ports produced `radius_fallback` at confidence 0.00, i.e. no
         enclosing vector path exists at all. There is no box around the text; it is bare text
         with a leader line.

      2. *A connector is where a pipe ends without reaching a symbol, so snap to the nearest
         `loose_end`.* Appeared to be confirmed 22/22 within 0.07-0.20in — **but that
         measurement was CIRCULAR and is retracted.** The graph it measured against had the
         port text boxes included in the symbol set, and the tracer MASKS symbol boxes, so
         lines were being cut at each text box's own mask boundary. 735 of the 1,749 loose ends
         existed only because of those boxes. The 25-72px "distances" were essentially each
         text box's own half-diagonal (a 110x20 box has a 56px half-diagonal) — the box
         measured against itself.
         Unconfounded (ports excluded from the traced symbol set): 1,014 loose ends, median
         distance **183px**, and only **8 of 22** within 108px. There is no tight structural
         relationship.

    So `extent` currently remains the annotation TEXT bbox, and boundary edges stay
    under-detected. `relocate_ports_to_loose_ends` is retained but is OFF by default — the
    machinery is sound, the evidence for using it is not.

    What IS verified: a candidate whose tag also exists as an on-sheet symbol is not a doorway
    at all. On this sheet exactly 3 of 22 candidates duplicate an on-sheet symbol
    (`MBD-0100`, `PBA-0201`, `HAM-0100`, all in a list region at the top of the left column) and
    those same 3 are precisely the 3 farthest from any pipe end (413px, 377px, 364px) — a clean
    separation with no overlap. They are filtered out in `classify_prod_tags`.

    `ref_tags` is a LIST, not a scalar: one physical connector routinely serves several remote
    destinations ("TO PBM-0450/0451"). Measured on the same sheet — `PBA-0501`/`PBA-0502`,
    `PBM-0450`/`PBM-0451` and `PBA-0903`/`PBA-0953` each resolve to a single shared loose end.
    Modelling those as separate ports would recreate the border-column artifact from the other
    direction, inventing a doorway that isn't there.
    """
    id: str
    extent: BBox
    ref_texts: List[str] = field(default_factory=list)   # raw annotation text(s)
    ref_tags: List[str] = field(default_factory=list)     # parsed equipment ids
    ref_sheet: Optional[str] = None          # parsed sheet number, if present
    direction: Optional[str] = None          # "in" | "out" | None (from the arrow)
    ref_conf: Optional[float] = None
    detect_source: str = "loose_end"
    anchor: Optional[Point] = None            # the loose end itself, for auditing
    anchor_dist_px: Optional[float] = None    # text centre -> loose end, for auditing

    @property
    def ref_tag(self) -> Optional[str]:
        """First reference, for callers that only need one (display, series checks)."""
        return self.ref_tags[0] if self.ref_tags else None

    @property
    def ref_text(self) -> Optional[str]:
        return self.ref_texts[0] if self.ref_texts else None

    def tracer_bbox(self) -> BBox:
        return self.extent

    def center(self) -> Point:
        x0, y0, x1, y1 = self.extent
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass
class LabelAnnotation:
    """Text that annotates. Never a node, never an endpoint. Kept because it's still useful:
    a pipe-spec label carries the line's spec, which R5 can use as evidence and which is the
    only remaining route to line-typing after PR #711 proved stroke-style dead on these CAD
    exports (everything flattens to solid single-width strokes, 0% dashed)."""
    id: str
    bbox: BBox
    text: str
    kind: str = "unknown"                   # line_spec | note | titleblock | unknown


@dataclass
class EntitySet:
    """What R0 hands to R1. `mode` records where the entities came from, and is carried all
    the way through to the score row — the three modes are never mixed in one number.

      gt_injected   — perfect entities from PID2Graph GT. Isolates relation quality; the
                      regression baseline every existing number was measured in.
      hand_verified — human-checked extents on real sheets, where no entity GT exists.
      detected      — Molmo2 / the multi-arm union. The honest end-to-end number.
    """
    symbols: List[SymbolNode] = field(default_factory=list)
    ports: List[PortNode] = field(default_factory=list)
    labels: List[LabelAnnotation] = field(default_factory=list)
    mode: str = "detected"

    # ── invariant checks — enforced, not documented-and-hoped-for ──────────────
    def validate(self) -> List[str]:
        """Returns a list of invariant violations (empty == clean). Cheap; call it in every
        pipeline entry point rather than trusting callers."""
        problems: List[str] = []
        seen: Set[str] = set()
        for n in list(self.symbols) + list(self.ports):
            if n.id in seen:
                problems.append(f"duplicate node id: {n.id}")
            seen.add(n.id)
        label_ids = {l.id for l in self.labels}
        if label_ids & seen:
            problems.append(f"label ids collide with node ids: {sorted(label_ids & seen)}")
        for s in self.symbols:
            if s.extent is None:
                continue
            x0, y0, x1, y1 = s.extent
            if x1 <= x0 or y1 <= y0:
                problems.append(f"{s.id}: degenerate extent {s.extent}")
            if s.point is not None:
                px, py = s.point
                if not (x0 <= px <= x1 and y0 <= py <= y1):
                    problems.append(f"{s.id}: seed point {s.point} outside extent {s.extent}")
            if s.extent_source not in EXTENT_SOURCES:
                problems.append(f"{s.id}: unknown extent_source {s.extent_source!r}")
        return problems

    # ── adapters into the existing tracer/builder API ─────────────────────────
    def tracer_inputs(self, *, include_ports: bool = True) -> Tuple[List[str], List[BBox], Dict[str, str]]:
        """(ids, bboxes, class_of) for `build_vector_page_graph` / `build_topology_relations`.

        Symbols contribute their resolved extent; ports contribute the doorway box at their
        loose pipe end. Labels contribute nothing — they are not endpoints. A symbol with no
        extent yet is skipped (R0 hasn't run, or failed on it) rather than silently falling back
        to a name plate, which is precisely the bug this contract exists to prevent.

        `include_ports=False` is the first of the two tracing passes: ports must be LOCATED
        before they can be traced against, and they are located by finding where pipes end
        loose. Including a provisionally-placed port in that first pass could absorb the very
        endpoint we need to stay loose, so pass 1 runs symbols-only.
        """
        ids: List[str] = []
        bboxes: List[BBox] = []
        class_of: Dict[str, str] = {}
        for s in self.symbols:
            bb = s.tracer_bbox()
            if bb is None:
                continue
            ids.append(s.id)
            bboxes.append(bb)
            class_of[s.id] = s.type or "other"
        if include_ports:
            for p in self.ports:
                ids.append(p.id)
                bboxes.append(p.tracer_bbox())
                class_of[p.id] = "port"
        return ids, bboxes, class_of

    def terminal_ids(self) -> Set[str]:
        """Nodes a walk may END at but must never pass THROUGH. Every port, always."""
        return {p.id for p in self.ports}

    def port_by_id(self) -> Dict[str, PortNode]:
        return {p.id: p for p in self.ports}

    def symbol_by_id(self) -> Dict[str, SymbolNode]:
        return {s.id: s for s in self.symbols}

    def extent_source_of(self) -> Dict[str, str]:
        """For score-time stratification by extent quality."""
        out = {s.id: s.extent_source for s in self.symbols}
        out.update({p.id: "port_extent" for p in self.ports})
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Legacy adapter — prod's flat tag list -> the v2 contract
# ──────────────────────────────────────────────────────────────────────────────

def classify_prod_tags(
    tags: Sequence[dict],
    *,
    page_size: Optional[Tuple[int, int]] = None,
    border_frac: float = 0.16,
    updown_frac: float = 0.10,
    mode: str = "hand_verified",
) -> EntitySet:
    """Split prod's `tags` array into symbols / ports / labels.

    This is how Phase 1 gets a v2 EntitySet without waiting on Molmo2: take the real
    extraction output, sort it into the three piles, then hand-correct the equipment extents
    (which is the only part that actually matters — see below).

    IMPORTANT, and the reason this adapter does NOT invent extents: prod's output carries no
    symbol-shape signal at all. Verified 2026-07-27 on PX-2368-0180004-001 — `symbols`,
    `elements` and `edges` are all empty arrays and `symbol_shape == "none"` on all 104 tags.
    So every `bbox_px` here is a TEXT bbox. We pass it through as `extent_source="given_bbox"`
    and flag it, rather than pretending it's a shape.

    Measured medians on that sheet make the split obvious:
        equipment      120x20 px   <- a whole vessel, as its name plate
        line           144x20 px   <- correct; it IS text
        instrument      58x42 px   <- plausibly the real ISA bubble
        valve           58x42 px   <- plausibly the real symbol
    Which is why all 5 measured wrong-endpoint failures have a wrong EQUIPMENT end and never a
    wrong instrument end: bubbles are boxed about right, equipment is not. Correcting the
    equipment extents alone should capture most of the win — roughly 6-8 shapes per cluster
    set rather than 30+.

    Port detection here is the deliberately-cheap first cut (proposal 3.1): off-page connectors
    sit in vertical columns just inside the sheet border by drafting convention, so an
    equipment-typed tag whose centre falls in the outer `border_frac` of page width is treated
    as a doorway. On PX-2368 that finds the two real columns at x~838 and x~5290.
    NOTE this is a heuristic on an unvalidated axis: Probe 3 (87.5%) validated *reading* a
    connector once you have its crop, NOT *finding* it. Every port carries
    `detect_source="border_heuristic"` so its contribution stays auditable.
    """
    symbols: List[SymbolNode] = []
    ports: List[PortNode] = []
    labels: List[LabelAnnotation] = []

    width = page_size[0] if page_size else None

    # A candidate doorway whose tag ALSO appears as an on-sheet symbol is not a doorway — it is
    # a schedule/equipment-list entry, or a duplicate label. Verified on PX-2368-0180004-001:
    # exactly 3 of 22 candidates duplicate an on-sheet symbol, and those same 3 are exactly the
    # 3 farthest from any traced pipe end (413/377/364px vs 78-170px for the rest) — a clean
    # separation, no overlap. Computed up front because the decision needs the whole tag list.
    height = page_size[1] if page_size else None

    def _in_side_margin(bb: BBox) -> bool:
        if width is None:
            return False
        cx = (bb[0] + bb[2]) / 2.0
        return cx < border_frac * width or cx > (1.0 - border_frac) * width

    def _in_updown_margin(bb: BBox) -> bool:
        """Top/bottom margin band. Separate from the side band and narrower, because the
        left/right edges carry off-page CONNECTORS (real nodes) whereas the top/bottom edges
        carry title blocks and equipment SCHEDULES (not nodes)."""
        if height is None:
            return False
        cy = (bb[1] + bb[3]) / 2.0
        return cy < updown_frac * height or cy > (1.0 - updown_frac) * height

    def _in_body(bb: BBox) -> bool:
        return not _in_side_margin(bb) and not _in_updown_margin(bb)

    def _is_text_shaped(bb: BBox, *, max_height: float = 35.0, min_aspect: float = 2.5) -> bool:
        """Wide-and-thin box signature of a text callout, vs. the squarish signature of a real
        drawn symbol/bubble. HEIGHT is the primary discriminator, not aspect ratio alone — a real
        ISA bubble stays >=~40px tall even at 3:1 aspect (measured 51-122 x 40-43px squarish
        records), while a text callout is always short (measured 106x20 and 153x31, both <35px
        tall). Aspect ratio alone would misclassify a wide-but-tall symbol at the boundary."""
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if h <= 0:
            return False
        return h < max_height and (w / h) >= min_aspect

    # Texts that appear on a real, body-region entity. Anything matching one of these but
    # sitting in a MARGIN is a list/schedule restatement of it, not a second instance.
    on_sheet_texts = {
        (t.get("text") or "") for t in tags
        if t.get("type") not in LABEL_TAG_TYPES
        and len(t.get("bbox_px") or []) == 4
        and _in_body(tuple(t["bbox_px"]))
    }

    for t in tags:
        tid = t.get("id")
        if not tid:
            continue
        bbox = t.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        bb: BBox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        ttype = t.get("type")
        text = t.get("text") or None

        if ttype in LABEL_TAG_TYPES:
            labels.append(LabelAnnotation(id=tid, bbox=bb, text=text or "", kind="line_spec"))
            continue

        # A margin-band tag whose text also appears on a body-region entity is a schedule /
        # specification-table restatement, never a node. Checked on ALL FOUR margins, not just
        # left/right — verified on PX-2368-0180004-001, where a horizontal specification table
        # runs across the TOP of the sheet ("XFMR-0301 / BULK OIL TRANSFORMER A / 125KVA 480V
        # 60Hz", rendered and confirmed to contain no symbol at all). Five tags live in that band
        # (y<400 of a 3960px sheet); the original left/right-only rule caught only the two that
        # happened to also be far left, and one of the survivors even resolved a spurious extent
        # off a table cell border — a false success that would have entered the node set.
        if text and text in on_sheet_texts and (_in_side_margin(bb) or _in_updown_margin(bb)):
            labels.append(LabelAnnotation(id=tid, bbox=bb, text=text, kind="schedule"))
            continue

        in_border = width is not None and ttype == "equipment" and _in_side_margin(bb)
        if in_border:
            # Located at the annotation TEXT. This is a KNOWN limitation, not a finished state:
            # pipes terminate at the connector graphic, not the text, so boundary edges stay
            # under-detected (measured: 1 edge across 22 candidates). Two attempts to relocate
            # were tested and refuted — see PortNode's docstring.
            ports.append(PortNode(
                id=tid, extent=bb,
                ref_texts=[text] if text else [],
                ref_tags=[text] if text else [],
                detect_source="border_text",
            ))
            continue

        symbols.append(SymbolNode(
            id=tid, extent=bb, extent_source="given_bbox",
            tag=text, tag_source="prod_tag", type=ttype, source_arm="prod_ocr_reasoning",
        ))

    # ── Duplicate-record dedupe: text callout vs. bubble (2026-07-28 fix) ────────────────
    # Same-text duplicate PRINTED RECORDS in the drawing body (not a schedule restatement,
    # already filtered above) — e.g. a PSV appears once as a text callout beside its tag AND
    # once as the actual ISA bubble symbol. Left un-deduped, both become separate SymbolNodes
    # sharing a tag, and the tracer routinely connects them (they sit close together), producing
    # a device-connected-to-its-own-label false positive — measured 1 of 7 on-sheet false
    # positives on PX-2368-0180004-001 (4 PSVs affected: PSV-0100A/B, PSV-0300A/C).
    # Rule: for a text shared by >=2 body-region records, drop the text-SHAPED one(s) only when
    # at least one symbol-SHAPED record for the same text also exists — shape-based (generalizes
    # across sheets), not coordinate-based. A text shared by two records that are BOTH
    # symbol-shaped (e.g. SDV-0100B on the same sheet, two squarish records 132px apart) is left
    # untouched — that is a genuinely ambiguous case (possibly two real, distinct valves), not a
    # callout/bubble duplicate, and this rule must not guess on it.
    by_text: Dict[str, List[SymbolNode]] = {}
    for s in symbols:
        if s.tag:
            by_text.setdefault(s.tag, []).append(s)
    drop_ids: Set[str] = set()
    for text, group in by_text.items():
        if len(group) < 2:
            continue
        text_shaped = [s for s in group if s.extent and _is_text_shaped(s.extent)]
        symbol_shaped = [s for s in group if s.extent and not _is_text_shaped(s.extent)]
        if text_shaped and symbol_shaped:
            drop_ids.update(s.id for s in text_shaped)
    if drop_ids:
        symbols = [s for s in symbols if s.id not in drop_ids]

    return EntitySet(symbols=symbols, ports=ports, labels=labels, mode=mode)


def relocate_ports_to_loose_ends(
    entity_set: EntitySet,
    loose_ends: Sequence[Point],
    *,
    render_dpi: int,
    max_snap_inches: float = 0.30,
    half_box_inches: float = 0.09,
) -> Dict[str, int]:
    """Move each port from its annotation text onto the loose pipe end it names, then dedupe
    ports that share one end. Returns a summary dict.

    This is the fix for boundary detection being effectively dead (1 edge from 22 ports). See
    `PortNode`'s docstring for the two hypotheses tested and why this one is right.

    `max_snap_inches` = 0.30 is deliberately just above the measured maximum (0.20 in / 72 px on
    PX-2368) rather than generously wide: every one of the 22 real cases fell in 0.07-0.20 in, so
    a tight bound admits the real population while refusing to invent a port for an annotation
    that has no pipe arriving at it. A text with no loose end within the bound keeps its
    provisional text-box location and is marked `detect_source="unsnapped"` — visible as a
    failure rather than silently producing a port that can never connect.

    Dedupe: ports snapping to the SAME loose end are merged into one, accumulating `ref_tags`.
    One connector serving two destinations is one doorway.
    """
    if not loose_ends:
        return {"snapped": 0, "unsnapped": len(entity_set.ports), "merged": 0}

    max_snap = max_snap_inches * render_dpi
    half = half_box_inches * render_dpi

    # loose end (rounded to whole px) -> the port that claimed it
    claimed: Dict[Tuple[int, int], PortNode] = {}
    kept: List[PortNode] = []
    snapped = unsnapped = merged = 0

    for port in entity_set.ports:
        x0, y0, x1, y1 = port.extent
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best = min(loose_ends, key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2)
        dist = ((best[0] - cx) ** 2 + (best[1] - cy) ** 2) ** 0.5

        if dist > max_snap:
            port.detect_source = "unsnapped"
            port.anchor_dist_px = round(dist, 1)
            kept.append(port)
            unsnapped += 1
            continue

        key = (int(round(best[0])), int(round(best[1])))
        if key in claimed:
            # Same physical doorway, another remote destination.
            owner = claimed[key]
            for t in port.ref_texts:
                if t not in owner.ref_texts:
                    owner.ref_texts.append(t)
            for t in port.ref_tags:
                if t not in owner.ref_tags:
                    owner.ref_tags.append(t)
            merged += 1
            continue

        port.extent = (best[0] - half, best[1] - half, best[0] + half, best[1] + half)
        port.anchor = (float(best[0]), float(best[1]))
        port.anchor_dist_px = round(dist, 1)
        port.detect_source = "loose_end"
        claimed[key] = port
        kept.append(port)
        snapped += 1

    entity_set.ports = kept
    return {"snapped": snapped, "unsnapped": unsnapped, "merged": merged}


SHEET_REF_RE = None  # set lazily in detect_ports_from_sheet_refs to keep `re` import local


def detect_ports_from_sheet_refs(
    pdf_path: str,
    tags: Sequence[dict],
    *,
    render_dpi: int,
    page_index: int = 0,
    page_size: Optional[Tuple[int, int]] = None,
    border_frac: float = 0.16,
    sheet_ref_pattern: str = r"^\d{7}-\d{3}$",
    pair_dy_inches: float = 0.45,
    size_tolerance: float = 2.0,
) -> Tuple[List[PortNode], Dict[str, object]]:
    """Locate off-page connector ports at the PENTAGON that encloses each sheet-number token.

    This is what the drawing convention actually is, established by rendering and *looking* at
    the border regions after two hypotheses failed:

        ───────────────── 2" - 245 PSIG ──────────────<  0180014-001  >
                            FROM GLYCOL COND.
                            SEPARATOR (MBD-0635)

    The pipe run terminates in a pentagon/flag containing the REFERENCED SHEET NUMBER. The
    equipment name sits in bare text BELOW the line, on the drawing side. Both borders use the
    identical convention, mirrored.

    That geometry explains both earlier failures precisely:
      - Seeding extent resolution from the EQUIPMENT text found no enclosing path (11/14
        `radius_fallback` at confidence 0.00) because that text genuinely has no box. Seeding
        from the SHEET-NUMBER token instead: **15/15 enclosed, all at confidence 1.00**, and
        14/15 resolve to a consistent **203x42 px** box — the drafted pentagon.
      - Snapping to loose ends "worked" at 22/22 only because the measurement was circular (see
        `PortNode`). The genuine unconfounded distances of 78-265px were never noise: they are
        the real offset from the equipment text to the pentagon.

    Pairing is tight and directional: the equipment text sits ~45px BELOW its pentagon
    (measured 43-45px on every checked case), on the drawing side. `pair_dy_inches` bounds it.

    `size_tolerance` guards the one measured outlier — a pentagon that resolved to 880x42 rather
    than 203x42, having merged with the pipe line. Anything more than `size_tolerance` x the
    median width is replaced by a median-sized box centred on the token, so an over-wide path
    cannot hoover up unrelated pipe endpoints.

    Returns (ports, stats).
    """
    import re
    import fitz

    from extent_resolution import extract_page_vector_paths, resolve_extent_from_seed

    pattern = re.compile(sheet_ref_pattern)
    zoom = render_dpi / 72.0
    paths, pg = extract_page_vector_paths(pdf_path, page_index, render_scale=zoom)
    if page_size is None:
        page_size = pg
    width = page_size[0]

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        rmat = page.rotation_matrix
        tokens: List[Tuple[str, BBox]] = []
        for w in page.get_text("words"):
            txt = (w[4] or "").strip()
            if not pattern.match(txt):
                continue
            r = fitz.Rect(w[0], w[1], w[2], w[3]) * rmat
            bb = (min(r.x0, r.x1) * zoom, min(r.y0, r.y1) * zoom,
                  max(r.x0, r.x1) * zoom, max(r.y0, r.y1) * zoom)
            cx = (bb[0] + bb[2]) / 2.0
            if cx < border_frac * width or cx > (1.0 - border_frac) * width:
                tokens.append((txt, bb))
    finally:
        doc.close()

    # Resolve each token's enclosing pentagon.
    raw: List[Tuple[str, BBox, BBox, str, float]] = []  # (sheet, token_bb, extent, src, conf)
    for txt, bb in tokens:
        seed = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        extent, src, conf = resolve_extent_from_seed(
            seed, paths, render_dpi=render_dpi, page_size=page_size)
        raw.append((txt, bb, extent, src, conf))

    # Median width guards the merged-path outlier.
    widths = sorted(e[2] - e[0] for _, _, e, _, _ in raw) or [0.0]
    med_w = widths[len(widths) // 2]
    med_h = sorted(e[3] - e[1] for _, _, e, _, _ in raw)[len(raw) // 2] if raw else 0.0

    oversized = 0
    ports: List[PortNode] = []
    for i, (sheet, tok_bb, extent, src, conf) in enumerate(raw):
        w = extent[2] - extent[0]
        if med_w > 0 and w > size_tolerance * med_w:
            cx = (tok_bb[0] + tok_bb[2]) / 2.0
            cy = (tok_bb[1] + tok_bb[3]) / 2.0
            extent = (cx - med_w / 2.0, cy - med_h / 2.0, cx + med_w / 2.0, cy + med_h / 2.0)
            src = "sheet_ref_median_box"
            oversized += 1
        ports.append(PortNode(
            id=f"port{i:03d}", extent=extent, ref_sheet=sheet,
            ref_conf=conf, detect_source=src if src != "vector_seeded" else "sheet_ref_pentagon",
            anchor=((tok_bb[0] + tok_bb[2]) / 2.0, (tok_bb[1] + tok_bb[3]) / 2.0),
        ))

    # Pair each pentagon with the equipment annotation text sitting just below it.
    dy_max = pair_dy_inches * render_dpi
    unpaired = 0
    for port in ports:
        px0, py0, px1, py1 = port.extent
        pcx, pcy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
        best = None
        best_d = None
        for t in tags:
            if t.get("type") in LABEL_TAG_TYPES:
                continue
            bb = t.get("bbox_px") or []
            if len(bb) != 4:
                continue
            tcx, tcy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
            if tcx < border_frac * width or tcx > (1.0 - border_frac) * width:
                pass
            else:
                continue
            dy = tcy - pcy
            if not (0 <= dy <= dy_max):
                continue
            d = abs(dy) + abs(tcx - pcx) * 0.25   # prefer directly below, mild x tolerance
            if best_d is None or d < best_d:
                best, best_d = t, d
        if best is None:
            unpaired += 1
            continue
        txt = best.get("text")
        if txt:
            port.ref_texts.append(txt)
            port.ref_tags.append(txt)

    stats: Dict[str, object] = {
        "sheet_ref_tokens": len(tokens),
        "pentagons_resolved": sum(1 for p in ports if p.detect_source == "sheet_ref_pentagon"),
        "oversized_replaced": oversized,
        "unpaired_with_equipment_text": unpaired,
        "median_pentagon_size": (round(med_w), round(med_h)),
    }
    return ports, stats


def loose_ends_from_graph(page_graph) -> List[Point]:
    """Every `loose_end` endpoint position in a traced page graph — the candidate set
    `relocate_ports_to_loose_ends` snaps against. A page has many (1,749 on PX-2368), which is
    why the annotation text is what makes the selection specific: loose ends alone are far too
    numerous to be ports on their own."""
    out: List[Point] = []
    for seg in page_graph.segments:
        for ep in (seg.endpoint_a, seg.endpoint_b):
            if ep.kind == "loose_end":
                out.append((float(ep.position[0]), float(ep.position[1])))
    return out


def from_gt_nodes(nodes: Dict[str, object], *, mode: str = "gt_injected") -> EntitySet:
    """Build an EntitySet from PID2Graph GT nodes (`pid2graph_gt.GTNode`: .bbox, .cls).

    GT bboxes ARE real symbol extents, so `extent_source="hand"` is honest here — this is the
    one input mode where extent quality is not in question. GT carries no tag text (PID2Graph
    is class-agnostic), so every node is an untagged symbol, which is exactly the case the
    contract is built to handle. No ports: PID2Graph sheets have no border connector columns
    at all, which is precisely why the 762-sheet corpus check never caught the border-column
    failure the backbone pass hit on real drawings.
    """
    symbols = [
        SymbolNode(
            id=nid, extent=tuple(getattr(n, "bbox")), extent_source="hand",
            tag=None, type=getattr(n, "cls", None), source_arm="pid2graph_gt",
        )
        for nid, n in nodes.items()
    ]
    return EntitySet(symbols=symbols, ports=[], labels=[], mode=mode)
