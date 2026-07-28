# HANDOFF — CLOSED

**This project is concluded (2026-07-28).** The final documentation is the research record in
[`RESEARCH/`](RESEARCH/00_INDEX.md) — read that first, not this file:

| | |
|---|---|
| [`RESEARCH/00_INDEX.md`](RESEARCH/00_INDEX.md) | Start here — one-paragraph verdict and navigation |
| [`RESEARCH/01_OBJECTIVE_AND_METHOD.md`](RESEARCH/01_OBJECTIVE_AND_METHOD.md) | Objective, hard rules, how the work was run across three sessions |
| [`RESEARCH/02_RESULTS.md`](RESEARCH/02_RESULTS.md) | Every measured number with its sample size and reliability tier |
| [`RESEARCH/03_WHAT_FAILED.md`](RESEARCH/03_WHAT_FAILED.md) | Failures and dead ends with diagnosed root causes |
| [`RESEARCH/04_TAKEAWAYS.md`](RESEARCH/04_TAKEAWAYS.md) | The 30 transferable lessons |
| [`RESEARCH/05_IF_RESUMED.md`](RESEARCH/05_IF_RESUMED.md) | State at close, unexecuted levers, traps to avoid |

Everything below is the working log kept while the project was live. It is retained as history and is
superseded by the research record wherever the two differ.

---

_This file supersedes older dated continuation docs. Read this first. It is kept continuously
updated (see `CLAUDE.md` → "Handoff discipline") — overwritten in place, not versioned by date._

---

## Where things stand right now (2026-07-27)

**Deep background:** `AI_Continuation_Document-27Jul2026-1500.md` (repo root) has the full
detailed account of the relationship-stage (Pipeline 3) benchmarking session — architecture,
what was built, probe results, the Tier-1 upgrades, the R4-degenerate finding, recall tracing.
Read it for depth; this file tracks what's changed/decided since, and what's still open.

Published artifact with the full visual writeup: https://claude.ai/code/artifact/3b4ea58f-f77f-4a6c-9188-23cd04ed8aa2

**Corrections made to that continuation doc this session (2026-07-27):**
- The doc flagged `benchmarks/r4_validation_results_2026-07-25.json` as unverified-pushed-to-HF.
  **Confirmed pushed** — downloaded it directly, `smoke_cap: 8`, matches the "kept 0/8 per sheet"
  result already reported. Not an open question anymore.
- The doc claimed the prior session's ephemeral scratchpad
  (`/private/tmp/claude-501/-Users-tomgeorge-pid-ml/10ffbddb-.../scratchpad/`) would be
  inaccessible from a fresh session, and listed the adjudication overlay PNGs and one-off scripts
  as **not recoverable**. **This was wrong as of 2026-07-27** — the directory is still fully
  intact and reachable via a direct Bash path from a brand-new session (confirmed via `ls`,
  including `adjudicate_crops.py`, `build_r4_bundle.py`, `run_backbone_open100.py`,
  `region_*.png` overlays, `px2368_overview.png`, `artifact_edit/pipeline3.html`). Same for the
  separate 762-sheet PID2Graph GT corpus in the `a852824c-...` session's scratchpad. **This is
  still just OS temp space, though — not durable.** Treat "still there" as a closing window, not
  a permanent fix. See "Time-sensitive" below.
- Not previously flagged anywhere: `git status` shows a second, unrelated body of
  modified-but-uncommitted tracked files (`Stage4_Benchmarking_Checklist.md`, `results.csv`, 9
  notebooks under `notebooks/stage4/` and `notebooks/all_vlm_stages_benchmarking/`) plus one
  staged new notebook (`PerStageV3_Stage13_Relation_vs_GPT55.ipynb`). This is leftover from the
  separate 2026-07-14 per-stage-v3-adapter thread (see memory `session_2026-07-14_per_stage_v3_writeup.md`),
  not this session's relation-bench work. Untouched, just noted so it doesn't get mistaken for
  new work or accidentally swept into a commit.

**New standing rule adopted this session:** this file. Added to `CLAUDE.md` ("Handoff
discipline" section) and to memory (`feedback_continuous_handoff_doc.md`) so it self-enforces
across future sessions.

---

## Time-sensitive

- ✅ **Done (2026-07-27).** Rescued the 10ffbddb-session scratchpad artifacts:
  - 10 one-off scripts (`adjudicate_crops.py`, `build_r4_bundle.py`, `probe2_gpt55_ceiling.py`,
    `run_agreement_diff_all3.py`, `run_backbone_multisheet.py`, `run_backbone_open100.py`,
    `run_real_extraction_partB.py`, `test_backbone_pass.py`, `test_backbone_real_gt.py`,
    `test_partition.py`) copied into the repo at
    `scripts/relation_bench_2026_07_25_rescued/` (uncommitted, per standing "only commit when
    asked" rule).
  - Everything else worth keeping (PX-2368 adjudication overlay PNGs, region overlay renders,
    `probe2_fixed_boxes_results.json`, `probe2_gpt55_ceiling_results.json`, the saved artifact
    HTML, a few sanity-check images) zipped and pushed to HF as
    `timthy45/pnid-extraction-datasets/benchmarks/rescue_bundle_2026-07-25.zip` — confirmed
    present on HF after upload (5,991,974 bytes).
  - Skipped re-uploading `preds_gpt55_partB_2026-07-24/*.json` — confirmed byte-identical
    (same sizes) to what's already at `benchmarks/extraction_2026-07-24/*.json` on HF.
  - Skipped `probe_bundle/`, `r4_bundle/`, `sheets/` dirs — already durable as
    `probe_bundle_2026-07-24.zip`, `r4_bundle_2026-07-25.zip`, `sheets/AG_PNID.zip` /
    `RIVE_LTTS_Sample.zip` on HF.
  - ✅ **Checked (2026-07-27):** the separate PID2Graph GT corpus in the `a852824c-...`
    session's scratchpad turned out to need no rescue — **its file contents are already gone**
    (771 directories, 0 files; an OS temp-clean already ran, this is the exact fragility risk
    this whole exercise was about). No data lost, though: the real copy is already durable on
    HF at `timthy45/pnid-extraction-datasets/pid2graph/PID2Graph.zip` (9.3 GB), confirmed via
    `HfApi`. Earlier framing of this as "still on disk, still time-sensitive" was wrong — it
    was already gone locally and safe on HF at the same time. Nothing further to do here.

## Decisions still awaiting Tom (do not act unilaterally on these)

- Whether to run the full 492-candidate R4 Qwen validation anyway, despite the recommendation to
  skip it (recommendation given, never explicitly re-confirmed by Tom).
- Symbol-extent resolution (Tier-1 #3) design direction — genuinely blocked, needs a real design
  call (connected-component clustering? raster flood-fill? permanent crude-bbox workaround?).
- Whether `PASSTHROUGH_TAG_TYPES = {valve, instrument, fitting, safety_device}` in
  `relationship_pipeline.py` is the right frozen mapping — flagged, not reviewed.
- Whether to wire the entity-extraction stage's Molmo2+`tag_matching.py` nearest-OCR-word design
  into the relationship-stage tests (currently they use prod's OCR/Vision tag extraction only,
  which carries label locations, not symbol shapes — the recurring bug source, see
  `AI_Continuation_Document-27Jul2026-1500.md` §5/§6).
- ✅ **Done (2026-07-27).** Wrote the 2026-07-25 findings into `Benchmark_Gaps_Register.md` as
  a new "Part D" section (backbone-pass corpus check + multi-sheet F1, R4-degenerate-in-pipeline
  result, PX-2368 recall trace + delta adjudication). Also added 4 new Group 2 rows (gaps 23-26)
  for the decisions below that were previously only in chat/the artifact, so they're now
  discoverable from the register itself: #23 (full 492-candidate R4 run go/no-go), #24
  (`PASSTHROUGH_TAG_TYPES` review), #25 (wiring backbone pass into `agreement_diff.py`'s AG/RIVE
  path), #26 (Molmo2+`tag_matching.py` wiring into relationship-stage tests).

## Not yet started

- GPT-5.5-low as an R4 arm (cheap, no GPU — "does a competent validator help" as its own
  question).
- Wiring the backbone pass into `agreement_diff.py`'s AG/RIVE path (currently PID2Graph-only).
- Recall-tracing the other 2 of 3 real sheets (only PX-2368 hand-traced so far).

---

## Pipeline 3 v2 — symbol-first input (proposed 2026-07-27, NOT built)

Full spec: **`Pipeline3_v2_Change_Proposal.md`** (repo root). Summary of the architectural shift:
Pipeline 3's input changes from "a list of all tags" (where `bbox_px` is the printed *text
label*) to "coordinates + tags of the *symbols*" from the multi-arm entity-extraction stage
(Molmo2 points + nearest-OCR-word pairing). Three node kinds replace the flat tag list:
`SymbolNode`, `PortNode` (off-page connectors — a NEW first-class node type), `LabelAnnotation`
(not a node).

Key reasoning established this session:
- **Tier-1 #3 (symbol extent) is unblocked by this, not solved-by-code.** Probe 1 PASSED at
  extent reconstruction while #3 was BLOCKED — the difference is Probe 1 was *seeded*. Molmo2's
  point is the seed.
- **Frontier models are WORSE at this task, contra intuition** — frozen 20-sheet Gupta,
  class-agnostic detection: Molmo2-points F1 **0.6276** vs GPT-5.5-low F1 **0.5125**. Molmo2 has
  a native pointing head; the others emit coordinates as text tokens. So a frontier model
  CANNOT serve as a ground-truth entity source (idea raised and rejected with data) — worse, its
  errors are systematically the tag-text-vs-symbol bug, i.e. the variable under test, so it would
  contaminate the control group and make the A/B read null for the wrong reason.
- **Off-page handling:** separate the connector *graphic* (real extent on this sheet, valid
  terminal tracer endpoint) from the *remote equipment* it names (no location here, becomes an
  attribute). This makes gap #22's ~19-20 previously-unverifiable claims/sheet scoreable, and
  splits work along proven capability: Qwen reads (Probe 3 PASS 87.5%), geometry traces.
- **Scoring discontinuity guard:** three permanent input modes (`gt_injected` / `hand_verified` /
  `detected`), tagged per row, never mixed — same principle as the pre/post-LLM split.

**Recommended first build: Phase 1 A/B** — hand-verify extents for PX-2368's 3 clusters (~30-35
entities; the 26-connection hand trace already exists as GT), rerun R1+R2a, rescore. Predicts
strict recall **58% → ~77%** (all 5 current WRONG-ENDPOINT failures are tag-text errors). No GPU,
no API spend, one variable, existing baseline. **Not started — awaiting Tom's sign-off on the
proposal's §8 open decisions.**

## Pipeline 3 v2 — IMPLEMENTED 2026-07-27 (code written, verified running, uncommitted)

Spec: `Pipeline3_v2_Change_Proposal.md`. All code lives in `src/relation_bench/`. Verified
against the real PX-2368-0180004-001 PDF + its HF extraction JSON. **Not committed.**

**New files:** `entities.py` (SymbolNode / PortNode / LabelAnnotation / EntitySet + prod-tag
adapter), `extent_resolution.py` (R0 seeded point→extent + PDF paths-with-bboxes),
`precision_audit.py` (the precision-adjudication harness).
**Modified:** `graph_construction/path_traversal.py` + `build_relations.py` (terminal ids,
single-hop cap, emit-time port↔port suppression, `stats_out`), `relationship_pipeline.py`
(`run_relationship_pipeline_v2`, V2_PASSTHROUGH_TYPES, label assert),
`hierarchy/priors.py` (`build_hierarchy_v2`, real-drawing type vocab),
`score.py` (`V2ScoreRow`, `score_v2`, on-sheet/boundary split, ENTITY_MODES),
`pdf_vector_extract.py` (page-size bug fix).

**Measured on PX-2368 (all verified live, not predicted):**
- ✅ **Phase 0 regression gate PASSES** — legacy `run_relationship_pipeline` still returns
  exactly **78** (original) and **52** (upgraded), matching the recorded r4_validation numbers.
- **Claims dropped 78 → 23** (on_sheet 22, boundary 1). Labels leave the node set (45 of 104
  tags) and 12 port↔port pairs are suppressed.
- **R0 works:** 32 of 37 symbols `vector_seeded`, 5 `radius_fallback`. Instruments/valves
  resolve to a consistent 68×68 (real drafted symbols); equipment is unreliable **because the
  seed is a name-plate centre** — confirms equipment needs hand-placed or Molmo2 seeds.
- **Containment prior now fires: 5 matches, was 0.** Parented total is coincidentally still
  13/37, but composition changed from 13 connectivity-nearest to 5 containment + 8
  connectivity. Prediction confirmed.
- **Wrong-endpoint fix is PARTIAL, and this is the important nuance:** the correct vessel edges
  `MBD-0100↔LSHL-0100` and `MBD-0100↔TSH-0100` are **now found** (direct `line_segment`
  evidence) where before they were absent — but the wrong `↔HAM-0100` edges **persist** via
  `path_through_4/5_segments`. So recall improves; the false positive is *not* removed. Exactly
  the "bigger target catches more pipes, including wrong ones" risk, now measured.

**Two bugs found by building this:**
1. **Page-size transposition** in `pdf_vector_extract` — `page.rect * rotation_matrix`
   double-rotates (page.rect is already rotation-applied), returning 3960×6120 instead of the
   true 6120×3960. Latent, not harmful: the only consumer reads `max(w,h)`, which is
   transposition-invariant. Fixed anyway. Point transforms via `rmat` are correct and unchanged
   (verified: embedded word "MBD-0100" maps exactly onto tag `t0002`'s bbox).
2. **port↔port edges survived terminality** — caught by the invariant check I added. Traversal-
   only terminality misses passes 0/1, which emit direct endpoint pairs; border frame/leader
   lines snapped two adjacent doorways together. Moved suppression to emit time; 12 caught.

**ISA-series cross-check** (new, `precision_audit.series_disagrees`): first cut flagged any
series mismatch → 5 flags, of which 2 were known-REAL connections (`PBA-0201↔PBA-0202`,
`NBK-0300↔PBA-0202`). Restricted to child-device→equipment only (equipment↔equipment mismatch
carries no information — different units are *supposed* to interconnect). Now **1 flag =
`PSV-0300A↔PBA-0201`, a known true error, with 0 false alarms** on the pairs we know.

### Follow-up fixes, same day

**Fix A — per-node precedence + distance cap: WORKS, verified exactly.**
Root cause of the persisting wrong edges was two structural gaps: (1) the **distance cap was
never ported** — `build_relations`' own docstring claims "Stage 11 + hop/distance caps" but only
the hop cap existed, and the register credits those caps with 91→20 by removing spurious
transitive links; (2) precedence was **per-pair, not per-node**, so LSHL-0100 could hold a
direct gap-0 edge to the vessel AND a 5-segment traversal edge to HAM simultaneously.
Fix: a child device (instrument/valve/safety_device) with direct pass-0/1 evidence of an
equipment host does not acquire other hosts by transitive walk — threshold-free, grounded in the
ISA one-host convention. Plus a 0.75in traversal distance cap.
Evidence: direct edges median gap **0px** (max 50px), traversal edges median **124px** (max
1767px). **All 6 known-truth checks now pass** — both correct vessel edges PRESENT, both wrong
HAM edges ABSENT, and both previously-flagged real pairs (`PBA-0201↔PBA-0202`,
`NBK-0300↔PBA-0202`) still PRESENT. Exactly 2 single-host drops, no collateral.

**Fix B — port relocation: EVIDENCE RETRACTED, shipped OFF by default.**
⚠️ The "22/22 loose ends within 0.2in" measurement that justified this was **circular**. It was
taken against a graph that *included the port text boxes*, and the tracer MASKS symbol boxes —
so it was measuring each text box against its own mask boundary (735 of 1,749 loose ends existed
only because of those boxes; a 110×20 box has a 56px half-diagonal, matching the "25-72px"
distances). Unconfounded: 1,014 loose ends, median distance **183px**, only **8 of 22** within
108px. Enabling relocation moved boundary edges from 1 to **0**.
`relocate_ports_to_loose_ends` is retained but `relocate_ports=False` by default.
**Port location remains genuinely unsolved** — two hypotheses tested, both refuted (enclosing
graphic: 11/14 got `radius_fallback` at conf 0.00, no enclosing path exists; loose-end snap:
above).

**What WAS salvaged from Fix B — a clean deterministic filter.** A border candidate whose tag
also exists as an on-sheet symbol is a schedule/list entry, not a doorway. Exactly 3 of 22
(`MBD-0100`, `PBA-0201`, `HAM-0100`) duplicate an on-sheet symbol, and those same 3 are exactly
the 3 farthest from any traced pipe end (413/377/364px vs 78-170px for the rest) — clean
separation, zero overlap. Now filtered to `LabelAnnotation(kind="schedule")`. Ports 22 → 19.

**Current PX-2368 state:** symbols=37, ports=19, labels=48. Relations **20** (on_sheet 20,
boundary **0**), violations 0. Drops: single_host 2, distance_cap 1, port↔port 12.
Phase 0 gate still exact (78 / 52).

### Fix B, third attempt — SOLVED. Boundary detection 0 → 7 edges

Solved by **rendering the border region and looking at it**, after two hypotheses failed. The
actual drafting convention:

```
    ───────────── 2" - 245 PSIG ─────────────<  0180014-001  >
                    FROM GLYCOL COND.
                    SEPARATOR (MBD-0635)
```

The pipe terminates in a **pentagon containing the referenced SHEET NUMBER**. The equipment name
is bare text ~45px BELOW the line, on the drawing side. Both borders, mirrored, identical.

That geometry explains both earlier failures exactly:
- Seeding extent resolution from the EQUIPMENT text found nothing (11/14 `radius_fallback`,
  conf 0.00) because that text genuinely has no box around it.
- Seeding from the **SHEET-NUMBER token**: **15/15 enclosed, all confidence 1.00**, 14/15
  resolving to a consistent **203×42 px** box — the drafted pentagon. Consistency across 14
  cases is the signature of a standard symbol, not a coincidence.
- The "too far" loose-end distances (78–265px) were never noise — they were the real offset from
  the equipment text to its pentagon.

New: `entities.detect_ports_from_sheet_refs()` — finds sheet-ref tokens in the border band,
resolves each one's enclosing pentagon, guards the one measured merged-path outlier (880×42 →
replaced with a median-sized box), and pairs each pentagon with the equipment text ~45px below
it (measured 43–45px on every checked case; 0 unpaired).

**One further bug this exposed, and it was only visible because drops are logged:** the distance
cap was killing every boundary edge (13 dropped, boundary read 0). A boundary edge is *inherently*
long — equipment deep in the drawing out to a doorway at the border — whereas the cap exists to
remove spurious transitive links *between on-sheet symbols*. Now exempt when either end is a port.

**Result on PX-2368:** ports 15 (all paired), **boundary 7** (was 0), on_sheet 20, total 27,
violations 0. Drops: port↔port 8, distance-cap 6, single-host 2. Phase 0 gate still exact (78/52).
All 6 known-truth on-sheet checks still pass.

Recovered off-page connections (plausible, **not yet adjudicated**): `MBD-0100`→MBF-0623,
`MBD-0100`→PBA-0903, `NBK-0300`→MBF-0500, `SDV-0100B`→MBF-0500, `FSV-0100B`→MBF-0500,
`FSV-0100B`→PBM-0450, `FSV-0300A`→HBG-0335. Consistent with the Fable adjudication's finding that
PSV relief lines exit to "TO LP FLARE SCRUBBER (MBF-0500)" — three of the seven land there.

### Chasing the 8 unconnected pentagons — SOLVED. Boundary 7 → 16, all 15 doorways connected

Diagnosed per pentagon rather than guessed. **All 15 were already reachable — none were broken.**
Two inherited caps were discarding them, and one new rule was needed.

1. **Hop cap.** Every pentagon reaches a real symbol, but needs **2–14 hops** (median ~9). The
   inherited `max_depth=8` silently dropped 8 of them. Same class of error as the distance cap: a
   boundary pipe crosses the whole sheet, while the cap was sized for symbol↔symbol relations. Now
   a separate `V2_TERMINAL_MAX_DEPTH = 16`, gated per edge class so on-sheet stays at 8.
2. **Naively raising the cap inflates instead of fixing.** At depth 14 boundary went to 30 edges
   and at 20 to 52 — each doorway picking up every symbol within range. But a doorway is the end
   of ONE pipe run, so it connects to the symbol that run reaches. Added a nearest-only rule for
   terminals — the same principle as the single-host rule for instruments, and threshold-free.
3. **A doorway must not lose the contest to another doorway.** First cut of the nearest rule left
   only 7/15 connected: for 7 pentagons the shortest path landed on *another pentagon* (4–7 hops,
   e.g. `port001↔port002`, `port003↔port009`) because the border-column structure links them, and
   port↔port is suppressed at emit — so those ports starved. The contest now counts only paths to
   real symbols. All 15 connect.

**Result on PX-2368:** **boundary 16** (was 0), **15/15 doorways connected**, on_sheet 20
(unchanged), total 36, violations 0. Drops: far-terminal 23, distance 6, single-host 2,
port↔port 8. Phase 0 gate exact (78/52). All 6 known-truth on-sheet checks still pass.

**Independent corroboration — not fitted.** 10 of the 16 boundary edges land on `MBD-0100`. The
Fable adjudication (2026-07-23, written before any of this code existed) had concluded by eye that
MBD-0100 *is* the hub many off-page connectors feed into, naming specific ones: "FROM GLYCOL
CONDENSATE SEPARATOR (**MBD-0635**)", "FROM LP DEGASSER (**MBD-4150**)". Pure geometry now finds
both of those, plus HBG-0905, MBF-0623, MBM-0400, MBF-0920, PBA-0903, PBA-0501. Separately the
adjudication recorded PSV relief lines exiting to "TO LP FLARE SCRUBBER (**MBF-0500**)" — three
edges land there (`NBK-0300`, `SDV-0100B`, `FSV-0100B`).

**Gap #22 is now unblocked.** The ~19–20 off-page claims/sheet that were declared structurally
unverifiable can be checked: 16 are recovered from geometry, and each carries the referenced sheet
number so the claim's named equipment can be compared against the doorway's own text.
**Not yet adjudicated** — these are geometry's answer, corroborated in part, not certified.

## Benchmark run 2026-07-27 — relationship stage isolated (extraction corrected)

Goal: treat extraction as correct so only relationship quality is measured. Artifacts in
`.../scratchpad/bench_px2368/` (**ephemeral** — worklist.json + 42 crops; push if needed).

**Extraction correction — 4 of 6 recovered, not 6.**
- The 4 pump-suction instruments ARE tagged `FSV-0201A`/`PSHL-0201B`/`FSV-0202A`/`PSHL-0202B`,
  with the number on a **second line inside the bubble** — which is why the PDF text layer shows
  only bare `FSV`/`PSHL` tokens and GPT-5.5 never emitted them. Found by rendering the pump
  region. R0 resolved all 4 at **67–68×68 at confidence 1.00**, matching every other ISA bubble.
- `SDV-0100F` and `SDV-0300A` are **not in the PDF text layer at all** (0 hits) — not locatable
  without a manual visual hunt. So this is "extraction corrected for 4 known misses", NOT 100%.

**Result: 41 symbols / 15 ports / 48 labels → 42 claims (26 on-sheet, 16 boundary), 0 violations.**

**3 of the 4 newly-added instruments immediately got their correct connection** — so those
misses were extraction, not relationship logic, exactly as predicted.

**The 4th exposes a real limitation of Fix A.** `PSHL-0202B` has a spurious *direct segment* edge
to `PBA-0201` (the wrong pump). The single-host rule then blocks the correct traversal edge to
`PBA-0202`. **The rule amplifies a tracing error**: when direct evidence is wrong, it locks the
error in and suppresses the right answer. Its sibling had no competing direct edge and resolved
fine. Worth revisiting — possibly gate the rule on ISA-series agreement.

**ISA-series check: 2 flags, both real errors, no false alarms** — it independently caught
`FSV-0202A↔PBA-0201` and `PSHL-0202B↔PBA-0201`, the two wrong-pump edges.

**Boundary edge verified visually.** Crop 005 (`MBD-0635`→`MBD-0100`) traces cleanly: pentagon →
`2"-245 PSIG` → joins `3"-245 PSIG` → joins the `16"-245 PSIG` header → into the vessel. Matches
the Fable adjudication's independent prose. Also corrects an earlier worry: this vessel's name
plate sits INSIDE its ellipse, so the bbox-centre seed landed in the shape and R0's 972×187
extent is genuinely the vessel body (blue box confirmed on the ellipse).

### ⚠️ The precision number is NOT yet valid — read this before quoting anything

Adjudicated 19 of 42 (45%): **17 real, 2 false**. Naive precision 0.895 — **but that is
meaningless**, because I marked as real exactly the claims appearing in the recorded
FOUND-CLEANLY list. That subset is enriched in true positives *by construction*, and the 23
unadjudicated claims are precisely the ones whose status is unknown, i.e. where false positives
would live.

**Honest bounds over all 42: precision ∈ [0.40, 0.95].** Too wide to be useful.
Narrowing it requires reviewing the 23 remaining crops (13 boundary, 10 on-sheet) — that is the
outstanding work, and it needs eyes that did not write the pipeline (register gap #3+5).

**Recall vs reconstructed GT: 16/21 = 76.2%** (recorded baseline 58% strict / 77% near-miss).
Still missing: `MBD-0100↔PSV-0100B`, `HAM-0100↔SDV-0100D`, `NBK-0300↔LSL-0300C`,
`NBK-0300↔SDV-0300B`, `PSHL-0202B↔PBA-0202`.
**CAVEAT:** this GT is *reconstructed from a prose summary in the artifact* — the original
26-connection list was never persisted to disk, so n=21 not 26. A clean recall number needs the
trace redone and SAVED this time.

## ⚠️ GENERALIZATION TEST 2026-07-27 — v2 is substantially fitted to PX-2368

Ran v2 on all 3 AG/RIVE dev sheets. **Tom's suspicion was correct.**

| mechanism | PX-2368 (dev sheet) | GD-B-540 | PX-2365 |
|---|---|---|---|
| sheet size | 17×11 in | 33×23 in | 42×30 in |
| embedded text words | 942 | **1** | 1665 |
| R0 extent success | **86%** (32/37) | **19%** (16/84) | **43%** (123/286) |
| port detection | 15/15 | **0 — total failure** | 25/25 found, 20 connected |
| boundary edges | 16 | **0** | 24 |
| distance-cap drops | 6 | 36 | **229** |
| single-host rule fired | 2 | 0 | 0 |
| invariant violations | 0 | **2** | 0 |

**Three root causes, all real:**

1. **R0's `seed_from_bbox_center` is the dominant failure.** It only works when a tag's name plate
   sits *inside* its symbol. On PX-2368 MBD-0100's plate happens to sit inside the vessel ellipse —
   **that was luck, not design.** Elsewhere it collapses to 19–43%. Diagnostic shows the seeds *do*
   have 1–2 enclosing paths, so they're being rejected by the size/area filter: GD-B-540 has fewer,
   larger path objects (6,920 paths for a sheet 3× bigger), i.e. its CAD exporter merges strokes
   into region-sized paths rather than per-symbol outlines — the same exporter difference that left
   it with 1 embedded text word.
2. **Port detection has a hard dependency on the embedded text layer.** GD-B-540 has 1 word → 0
   sheet-ref tokens → 0 ports → 0 boundary edges. Fix B simply does not run on text-outlined PDFs.
3. **Absolute-inch constants don't scale.** The 0.75in distance cap is 4.4% of PX-2368's width but
   only 1.8% of PX-2365's — proportionally 2.4× tighter, which is why it dropped **229** edges there
   vs 6 on the dev sheet. Same for `pair_dy_inches=0.45` (7 pentagons left unpaired on PX-2365).

**Two invariant violations on GD-B-540**: `seed point outside extent`. Cause is a tolerance
mismatch — `resolve_extent_from_seed` tests containment with `pad=3.0` but `EntitySet.validate()`
requires strict containment. Cosmetic, but it means the resolver can return a box that doesn't
contain its own seed.

**What DID generalize:**
- Port detection on sheets *with* a text layer: **25/25 pentagons on PX-2365**, a different drawing
  family. The mechanism is sound where its precondition holds.
- The median-based pentagon size guard adapted correctly (203×43 → 468×107) — deriving the size from
  the sheet rather than hardcoding it was the right call.
- Label/schedule filtering, and the pipeline ran without crashing on all three.
- The single-host rule **never fired** on the two new sheets, so it is neither validated nor
  refuted — still n=1 evidence.

**The key reframe:** the largest failure is precisely the job **Molmo2** is meant to do. The
bbox-centre seed was explicitly labelled in `extent_resolution.py` as "NOT a substitute for a real
pointing model" — this test proves that caveat was load-bearing, not boilerplate. v2's architecture
isn't refuted; v2 *without a pointing model* is, and PX-2368 masked it by luck.

### CORRECTION to the above diagnosis (same day, after Tom pushed back)

I initially framed these failures as "tag dependence." **That framing was wrong.** Verified by
direct test on PX-2368: strip every tag name and every port reference, and the output is
**byte-identical** — 20 on-sheet, 16 boundary, 0 edges differing in either direction.
**v2's topology is fully tag-name-independent, exactly as designed.** Names are metadata.

The real dependency is narrower: **the seed comes from a tag's bbox CENTRE** (37/37 symbols) — not
the name, the *location*. A tag bbox is a name-plate location, so this is an **entity-source**
problem, not a tag-name problem. The GD-B-540 result therefore does **not** indict v2's design; it
indicts using prod's tag list as the entity source, which the code already labelled a stopgap.

Also correcting the port claim above: the text-layer dependency is in the IMPLEMENTATION, not the
concept. Port *location* (where a pipe terminates at a doorway) is pure geometry — the pentagon
graphic still exists on GD-B-540 as vector paths, only its text was outlined, so a shape-based
detector could find it. Port *reference* ("the far side is MBD-0635") needs text, but that is
metadata, not topology, and Probe 3 reads it at 87.5% from a crop. Those two were conflated.

**Consequence for next steps:** the priority is NOT patching constants — it is feeding the pipeline
real symbol points (Molmo2, or hand-placed points for a controlled test). Evidence that this is the
binding constraint: the 4 hand-seeded pump instruments resolved at **confidence 1.00**, because
their seeds were correct. The distance-cap scaling defect (229 drops on PX-2365) remains real and
worth fixing, but is secondary.

## Hand-seed work 2026-07-27 (option 1) — partial: PX-2368 done, 2 sheets outstanding

**Key discovery that reshaped the task: hand-placing SEEDS would have been wasted effort.** The
diagnostic showed the dominant R0 failure is not a bad seed — it is `MAX_SYMBOL_INCHES = 4.0`
rejecting every enclosing path (64/68 on GD-B-540, 163/163 on PX-2365), because those sheets' CAD
exporters group strokes into region-sized path objects. No seed can conjure a tight path that isn't
there. R0 success by type also showed the value is concentrated in **equipment** (instruments/valves
already arrive at ~58×42, roughly their real symbol; equipment arrives as a ~120×20 name plate).

### Two generalizable code fixes (no hand work, apply to every sheet)

1. **`given_bbox` fallback** — when R0 finds no plausible enclosing path, fall back to the entity's
   supplied bbox instead of an arbitrary 0.44in circle. A measured text/symbol box beats a guess.
   Eliminates `radius_fallback` entirely on PX-2368.
2. **Schedule filter extended to ALL FOUR margins** (was left/right only). PX-2368 has a horizontal
   **specification table across the top** — rendered and confirmed to contain no symbols ("XFMR-0301
   / BULK OIL TRANSFORMER A / 125KVA 480V 60Hz"). Five tags live in that band; the old rule caught
   only two, and one survivor had even resolved a spurious extent off a table-cell border — a false
   success that entered the node set. Now 6 schedule entries filtered, symbols 37 → 34.

### Hand extents: 3 boxes on PX-2368, read off a 200px coordinate grid

Stored in `src/relation_bench/hand_extents/px2368.json` (`t0029` XFMR-0301, `t0083` PBA-0201,
`t0084` PBA-0202). Known-truth on a **12-check** set: **11/12**.

**Honest caveat — hand extents are themselves a tuning knob.** My first (tight) boxes *lost two
real edges*; three iterations were needed:

| box variant | known-truth |
|---|---|
| tight (circle + base only) | 10/12 — lost `PBA-0201↔PBA-0202` and `NBK-0300↔PBA-0202` |
| wide (+ nozzle triangles) | 10/12 |
| wider (+ both connections) | **11/12** — restored `PBA-0201↔PBA-0202` |

So "hand-placed" is not automatically correct; a too-tight extent silently drops real connections.

**Also fixed by the correct pump extent:** `PSHL-0202B↔PBA-0202`, the 4th pump instrument that was
previously blocked by the single-host rule amplifying a spurious direct edge to the wrong pump.
All 4 pump-suction instruments now connect correctly (was 3/4).

### The distance cap should probably be removed

Swept 0.75in / 2in / 6in / disabled on PX-2368: known-truth is **11/12 at every setting**, and
distance drops go 7 → 2 → 0 → 0. So it has **no measured benefit here** — while dropping **229**
edges on PX-2365. A liability with no demonstrated upside. (It also did *not* cause the remaining
`NBK-0300↔PBA-0202` failure — that was my hypothesis and the sweep refuted it; cause still unknown.)

### Outstanding
- **GD-B-540 (4 equipment boxes) and PX-2365 (5) not yet done** — only PX-2368 was hand-boxed.
- `NBK-0300↔PBA-0202` unexplained (not the distance cap).
- Re-run the generalization test on all 3 sheets once their boxes exist.

## Review page built 2026-07-27 — adjudication NOT completed

**Deliverable, durable:** `benchmarks/review_px2368_2026-07-27.zip` on HF (6 MB) — 42 annotated
crops + `worklist.json` + `review.html`. Unzip, open `review.html`, judge with keys **1** (real) /
**2** (not real) / **3** (unsure), then "Download verdicts JSON". Generator kept at
`scripts/build_review_page.py`. Regenerated against the CURRENT config (hand extents + all fixes),
so the earlier crop set is stale.

**Claim mix (42):** 16 boundary-traversal, 14 on-sheet-direct, 12 on-sheet-traversal.

**⚠️ I did NOT adjudicate all 42.** I viewed 3 crops. Each full-resolution P&ID crop is expensive to
read, and I would rather report a sample of 3 than imply a review of 42. **Precision therefore
remains bounded, not measured.** The conflict flagged in register gap #3+5 also still stands — I
wrote this pipeline, so my verdicts are a third-signal opinion, not ground truth.

**New defect class found by the sample — entity duplication.** Crop #24 `PSV-0300C ↔ PSV-0300C`:
RED box is the text callout (`PSV-0300C` above a `180020-002` sheet-ref box), BLUE is the actual PSV
bubble. **The pipeline related a device to its own label.** Both are separate tag records in the
drawing body, so the margin-based schedule filter cannot catch them.

Surveying all duplicate tag texts (14 on this sheet) splits them cleanly:
- **6 schedule/spec duplicates** — already filtered by the all-margins rule ✓
- **3 off-page pairs** (`MBD-0635`, `MBF-0500`, `MBM-0400` appearing on both borders) — legitimately
  two distinct doorways ✓
- **4 PSVs with BOTH a text-callout and a bubble record** (`PSV-0100A/B`, `PSV-0300A/C`). Signature
  is the same aspect test already measured elsewhere: **106×20 wide-and-thin = text** vs
  **51–122×40–43 squarish = symbol**.
- 1 ambiguous (`SDV-0100B`, two squarish records 132px apart — possibly two real valves)

**Direct precision impact on this sheet is small — 1 claim of 42** — so a dedupe fix is worth doing
for correctness but will not move the number much here. Proposed rule: for same-text records, drop
the text-shaped one and keep the symbol-shaped one (shape-based, not coordinate-based, so it
generalizes).

**Accuracy status unchanged:** recall **12/15 = 80%** vs independent GT (recorded 2026-07-25, before
this session's work); precision **unmeasured**, bounded **[0.38, 1.00]**; **no F1 available**.
42 claims cover 15 known connections — which is exactly why precision is the missing half.

## Three Tier-2 fixes implemented + rerun 2026-07-28 — project being wound down, documented as-is

**Context:** Tom confirmed this project is being closed out; today's ask was the last 3 planned
fixes wired in, one rerun on the same PX-2368 benchmark, results documented as they land — no
further chasing beyond this. All 3 are code, verified running on the real sheet, **not
committed**.

**1. Child<->child suppression, shared-host qualifier.** `graph_construction/build_relations.py`
— new post-filter after all 3 passes. Root cause of 5/7 measured on-sheet false positives: a
direct/traversed edge between two child-class devices (instrument/valve/safety_device) with no
shared equipment host. Kept a blanket ban OFF the table on purpose — PID2Graph's own valve|valve
stratum is the strongest scored one (F1 0.314) — so the rule only fires when
`single_host_child_classes`/`host_classes` are both supplied (same gate the existing single-host
rule uses), and only drops a child<->child pair when the two ends' (already-known) hosts differ
or are unknown; a pair sharing one host survives. Every PID2Graph self-test that doesn't pass
these kwargs is untouched. New stat: `dropped_child_child_no_shared_host` (also threaded onto
`V2Result.dropped_child_child`).

**2. Inline-symbol association — `passes_through_symbols` was dead on the vector path.**
`line_tracing/vector_graph.py::build_vector_page_graph` never set this field (0 of 8,105
segments across the earlier corpus check), so `build_relations`' pass 0 (inline-chain) has been
dead code the whole time the vector fast-path has existed. Root cause: CAD draws one continuous
line straight through an inline valve/instrument (unlike the raster path, which masks the symbol
and needs `bridge_across_symbols` to reconnect it) — the vector builder had nothing that noticed
a finished segment's polyline crosses a symbol's bbox without splitting a node there. Added
`_resolve_passthrough_symbols` (Liang-Barsky segment/rect intersection, excludes a segment's own
resolved endpoints) and wired it into the final `Segment(...)` construction.

**3. Duplicate-record dedupe — callout text vs. bubble.** `entities.py::classify_prod_tags` —
post-pass over same-tag symbol groups. Root cause of 1/7 false positives: some tags (4 PSVs on
PX-2368) print TWICE in the drawing body — once as a text callout, once as the real ISA bubble —
and both entered the node set as separate SymbolNodes sharing a tag, so the tracer routinely
connected a device to its own label. Shape-based rule (not coordinate-based, so it generalizes):
height is the discriminator (a callout is always short, <35px tall; a real bubble is >=~40px even
at wide aspect ratios) — when a tag has exactly one text-shaped record and at least one
symbol-shaped record, drop the text-shaped one(s). A tag with only squarish duplicates (e.g.
`SDV-0100B`) is left alone — genuinely ambiguous, possibly two real devices, not this rule's job
to guess.

### Rerun on PX-2368-0180004-001, all 3 fixes wired in

Reproduced every part of yesterday's pipeline that is CODE: real GPT-5.5-low extraction (104
tags) → `classify_prod_tags` (now with the dedupe fix) → R0 (`seed_from_bbox_center` +
`resolve_extents`) → the 3 hand-verified equipment extents (`hand_extents/px2368.json`) →
`detect_ports_from_sheet_refs` → `run_relationship_pipeline_v2` (now with the child<->child and
passes_through fixes active). **NOT reproduced:** yesterday's one-off manual "fix00-03"
pump-instrument text-correction patch (4 tags: FSV-0201A/PSHL-0201B/FSV-0202A/PSHL-0202B on a
second bubble line) — its exact bbox coordinates were only ever described in prose during the
interactive session, never saved to a script or file, so inventing coordinates would not be an
honest rerun. This means today's symbol count (30) is lower than yesterday's final corrected
count (41) by design, not regression — those 4 (plus the 2 sheet-refs never in the text layer at
all, `SDV-0100F`/`SDV-0300A`) are simply absent from today's input, same as the pipeline's
earlier, pre-correction state.

**Verified directly, not just by relation count:**
- Dedupe: raw extraction has 23 duplicate-text tag pairs; after `classify_prod_tags`, only
  `SDV-0100B` (2 squarish records, correctly left ambiguous) remains duplicated among symbols.
  All 4 PSVs (`PSV-0100A/B`, `PSV-0300A/C`) now resolve to exactly one record each — the bubble.
- passes_through_symbols: 194 of 7,657 vector segments now carry a symbol reference (0 before).
- Child<->child: correctly dropped `PSV-0100B(safety_device)<->PSHL-0100` and
  `PSV-0100B<->LSL-0100B` (a safety device wrongly tied to nearby instruments with no shared
  host) while `LSHL-0100<->TSH-0100` survived (both share host `MBD-0100` directly).
- The 2 previously-fixed wrong `HAM-0100` edges (`LSHL-0100`/`TSH-0100`->`HAM-0100`) are still
  correctly ABSENT — Fix A's single-host rule is undisturbed by the new filters
  (`dropped_single_host` still shows exactly `[(HAM-0100,LSHL-0100), (HAM-0100,TSH-0100)]`).

**Result on this input:** symbols=30, ports=15, labels=51. Relations: **on_sheet=17,
boundary=16, total=33**, violations=3 (pre-existing `hand`-extent/seed-tolerance mismatch, same
cosmetic issue noted 2026-07-27 on GD-B-540, not new). Drops: distance_cap 6, single_host 2,
far_terminal 23, child_child (NEW) 2, port<->port 8. 19 of 30 symbols (63%) now appear in at
least one relation, up from the systematic exclusion the dead `passes_through_symbols` field
caused — 11 remain isolated (`BDV-0100A`, `FSV-0101`, `FSV-0300B/C`, `PBA-0202`, `PSV-0100B`,
`PSV-0300C`, `SDV-0100B/C/D`, `SDV-0300B`), a real, visible partial result, not fully resolved.
**Not compared to a P/R/F1 number** — the 21-connection reconstructed GT from 2026-07-27 was
itself flagged there as prose-reconstructed and never saved to disk, and re-deriving it from
scratch is out of scope for this closing pass. Full relation lists (by tag) and the raw JSON are
in `/tmp` scratch (`px2368_rerun_2026-07-28.json`) — ephemeral, not pushed anywhere durable;
rerun the script (`rerun_px2368_with_3fixes.py`, also ephemeral scratch) if needed again before
that path gets cleaned.

**Nothing further planned beyond this** — per Tom, this closes out the active work on this
benchmark; treat the above as the final state to report, not a checkpoint to keep building on.
