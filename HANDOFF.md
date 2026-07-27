# HANDOFF — current running state

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

## Immediate next action

All three of this turn's asks (scratchpad rescue, register write-up, PID2Graph corpus check)
are done — see above. Nothing currently in flight. Next candidates, in no particular order, all
awaiting Tom's pick:
- Any of Group 2 gaps 23-26 in `Benchmark_Gaps_Register.md`.
- The other not-yet-started items listed above (GPT-5.5-low R4 arm, backbone→agreement_diff
  wiring, recall-tracing the other 2 sheets).
