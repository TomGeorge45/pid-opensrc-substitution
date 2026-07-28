# Pipeline 3 v2 — final benchmark, 2026-07-28

Closing measurement for the relationship-stage work. The three remaining fixes were wired in, two
defects found during the rerun were corrected, and the benchmark was rerun on the same sheet.
Recorded as measured, including what did not work.

**Scope:** one sheet, `PX-2368-0180004-001` (real vector PDF, 360 dpi). Hand-assisted — 3 equipment
extents hand-placed, 4 missing entities hand-recovered. Deterministic half only (R0/R1/R2a); the LLM
stages were not run, and R4 is separately measured as degenerate (kept 0/8 on every sheet).

---

## Results

| Metric | 2026-07-27 | 2026-07-28 final |
|---|---|---|
| Claims | 42 | **42** (on-sheet 26, boundary 16) |
| Precision (adjudicated verdicts) | 82.1% | **93.9%** |
| Recall (15-edge independent GT) | 80.0% | **86.7%** |
| **F1** | 0.810 | **0.902** |
| Isolated symbols | 13 / 38 | **9 / 34** |
| Contract violations | 0 | **0** |

Precision matches each claim against the 42-claim adjudication saved on 2026-07-27:
**31 carried REAL, 2 carried FALSE, 3 carried UNSURE, 6 new and unadjudicated.**
Bounds including everything unresolved: **[73.8%, 95.2%]**.

**6 of the 7 known false positives are gone**, with **no adjudicated-real edge lost** (31 before and
after). The survivor is `FSV-0202A↔PBA-0201` — pump B's instrument bound to pump A. That is a
child-to-equipment edge, outside the child↔child rule's scope. The ISA tag-series check already flags
it automatically, but only as a ranking hint rather than a suppression.

**Recall gained one edge** (`MBD-0100↔PSV-0100B`). Still missed: `HAM-0100↔SDV-0100D` and
`NBK-0300↔SDV-0300B` — both inline valves that remain isolated in the traced graph.

---

## The five changes

| Change | Effect |
|---|---|
| Duplicate-record dedupe (text callout vs bubble) | symbols 38 → 34; removed the `PSV-0300C↔PSV-0300C` self-edge |
| Inline-symbol association, in the tracer (`_resolve_passthrough_symbols`) | populates `passes_through_symbols`, reviving Pass 0 |
| Inline symbols as routable traversal waypoints (`route_inline`) | isolated symbols 11 → 9; recall 80.0% → 86.7% |
| Child↔child suppression, **source-based** (rewritten — see below) | 6 of 7 false positives removed |
| Seed/extent tolerance fix | violations 3 → 0 |

The last two were corrections to work that was wrong on first attempt.

---

## Two things that were wrong and were redone

**The child↔child rule keyed on the wrong signal.** The first implementation asked whether the two
child devices shared a host equipment — keeping same-host pairs, dropping different-host ones. That
is measurably backwards: `LSHL-0100↔TSH-0100` and `PSHL-0100↔LSL-0100B` are two instruments hanging
off the *same* vessel by separate stems with no pipe between them. They share a host, so they
survived. It removed only 4 of the 7 false positives.

Host relationship is not the discriminator. What separates a real valve-in-series pair from two
instruments on one vessel is whether **a single pipe run physically passes through both** — which is
precisely what Pass 0 (inline-chain, sourced from `passes_through_symbols`) means. The rule now keeps
a child↔child edge only with Pass 0 evidence. Pass 1 (endpoint resolution, the mechanism that merges
two adjacent bubbles' stems into one segment) and Pass 2 (transitive traversal along a shared header)
are exactly the two failure modes. Inline valve↔valve runs — the strongest scored stratum on
PID2Graph, F1 0.314 vs 0.126 for asset↔asset — are preserved because they come from Pass 0.

**A seed/extent tolerance mismatch.** `resolve_extent_from_seed` tested containment with a 3px pad
while `EntitySet.validate()` required strict containment, so a resolved extent could fail to contain
its own seed (2–3 violations on GD-B-540 and PX-2368). The extent is now expanded minimally to
include the seed, rather than loosening the invariant. Violations went to 0.

---

## What to distrust

**The Phase 0 regression gate no longer holds, for an understood reason.** Legacy
`run_relationship_pipeline` returns **221 / 95** where it returned **78 / 52** before. Cause: the
tracer now populates `passes_through_symbols`, so Pass 0 fires on the legacy path for the first time
(6,235 of 9,784 segments; 191 of the 237 legacy relations come from Pass 0). This is a direct and
expected consequence of the inline fix, not a regression — but it does mean **refactor neutrality can
no longer be claimed**, and any "the v1 baseline was X" statement must be re-derived rather than
quoted.

**A retracted claim.** While debugging the above I asserted that `passes_through_symbols` "was always
populated and the earlier 0-of-8,105 measurement was taken on the wrong graph." That is wrong. The
original zero was correct at the time; the field became populated because the fix landed. Pass 0
genuinely *was* dead code on the vector path before 2026-07-28.

**6 new claims are unadjudicated**, and 4 look like the sibling pattern that the rewritten rule is
meant to catch: `PSHL-0300C↔LSL-0300C`, `FSV-0201A↔PSV-0300A`, `PSHL-0201B↔PSV-0300A`,
`PSV-0300A↔PBA-0201`. If they are false, precision moves toward the lower bound.

**Two reruns of this benchmark exist and disagree by design.** A parallel session's rerun reports
symbols=30, total=33 relations — it excluded the 4 hand-recovered pump instruments because their
coordinates were never saved to a file. This run includes them (symbols=34). Neither is wrong; they
measure different inputs. The 33-relation figure is the more conservative and the more reproducible,
since this one depends on coordinates that existed only in a chat transcript.

**Generalisation is untested.** Every number here is one sheet. On the other two dev sheets R0's
extent resolution drops to 19% and 43%, and port detection fails entirely on the sheet whose CAD
export outlines its text into vector paths (1 embedded word). Nothing here shows the result transfers.

**Self-adjudicated.** Verdicts came from the same party that wrote the pipeline — a third-signal
opinion, not certified ground truth. The recall GT is a 15-edge partial reconstruction of a trace that
was never persisted; the original 26-connection list is lost.

---

## Where this leaves the stage

On one real vector sheet, with entity extraction corrected by hand, the deterministic half of
Pipeline 3 v2 reaches **F1 ≈ 0.90** on adjudicated claims — up from 0.81. Boundary (off-page) edges
were the strongest component throughout, at 16/16 correct, and were the part that went from
structurally impossible to fully working.

Against the only multi-sheet benchmark with real ground truth, PID2Graph, the pipeline sits at
F1 **0.225** versus a trivial nearest-neighbour floor of 0.226 — but that is the raster path on a
corpus with a documented tracing ceiling, and is not comparable to the vector numbers above.

The honest summary: the approach works well on the sheet it was developed against, has one clearly
identified remaining defect (wrong-host binding, already auto-flagged by the ISA series check), and
has not been shown to generalise beyond that sheet.
