# PID-ML — research record

Final documentation for this repository. The work is concluded; nothing here is a plan for
further activity.

**What this repo was.** An attempt to replace every cloud LLM/VLM/OCR call in Rive's working P&ID
document-intelligence agent with a local, fine-tuned counterpart, and to prove the local version
matched the original. A substitution project, not a redesign.

**What actually happened, in one paragraph.** Over roughly three weeks the project built and
measured a great deal: a Stage-4 detection bake-off across four candidate VLMs, a domain-adaptation
LoRA (two failed generations, then a working per-stage design), benchmarks for four pipeline stages,
an end-to-end harness driving the *real* production agent code, a multi-arm entity-extraction
architecture, and a complete relationship-extraction pipeline that reached F1 ≈ 0.90 on its
development sheet. It also established, with numbers, that the cloud incumbent it was trying to
replace **does not itself clear the project's own pass bar** (Claude F1 0.380 on real detection vs a
0.70 bar), that no commercially-usable P&ID training corpus exists, and that general mixed-task
fine-tuning of a VLM is actively destructive. **The project's primary stated objective — select a
shared base VLM at Stage 4 and prove parity — was never reached.** The Stage-4 master gate has zero
boxes ticked and `base.md` was never filled in.

---

## The documents

| File | What it covers |
|---|---|
| [`01_OBJECTIVE_AND_METHOD.md`](01_OBJECTIVE_AND_METHOD.md) | What the project set out to do, the hard rules it held itself to, and how the work was actually conducted — including the three-session continuity method and the infrastructure it ran on |
| [`02_RESULTS.md`](02_RESULTS.md) | Every measured number, by stage, with its sample size and reliability caveat |
| [`03_WHAT_FAILED.md`](03_WHAT_FAILED.md) | Failures, dead ends and abandoned approaches, each with its diagnosed root cause |
| [`04_TAKEAWAYS.md`](04_TAKEAWAYS.md) | The transferable lessons — the things worth carrying to any similar project |
| [`05_IF_RESUMED.md`](05_IF_RESUMED.md) | Exact state at close, the highest-leverage unexecuted levers, and the traps a successor would otherwise re-walk into |

## A note on identifiers

Customer drawing identifiers and equipment tags are **pseudonymised** throughout these documents.
Sheets appear as `SHEET-1` … `SHEET-6`, equipment by role (`VESSEL-1`, `TREATER-1`, `PUMP-A`),
and the two source-sheet families as `Family-A` / `Family-B`. Numbering is stable across all six
files, so cross-references still line up, and every technical finding reads the same as it did with
the real names.

Two consequences worth knowing. The real identifiers still appear in the *working* documents this
record was built from (`Benchmark_Gaps_Register.md`, `Pipeline3_v2_*.md`, `HANDOFF.md`), in the code
and its comments, and in the committed adjudication and hand-extent JSON — de-identification was
applied to this record, not to the whole repository. And the reproduction script takes the sheet
identifier as an argument rather than hardcoding it, so you must supply the real stem to run it.

## How to read the numbers

Three reliability tiers are used throughout, and they matter more than the figures:

- **Measured** — a real run, adequate n, recorded config. Trustworthy.
- **Directional** — a real run at small n (often n=6 to n=25). Indicative only. This project has a
  documented case of an n=25 result reversing entirely at n=120.
- **Uncertified** — referenced in a document but the figure was never re-verified, or the
  ground truth it was scored against is constructed/reconstructed rather than real.

Where a number is uncertified, it says so inline. Nothing has been rounded up into a stronger claim
than its evidence supports, and several figures reported confidently in earlier session write-ups
are corrected here.

## Provenance

Primary sources, all in the repo root: `PID_Local_Substitution_Spec.md` (the original spec),
`Stage4_Benchmarking_Checklist.md` and `Stage4_Checklist_Status.md`, `Agent_Pipeline_Facts.md`
(code-verified facts about the real agent), `Benchmark_Gaps_Register.md` (the 26-gap analysis and
Parts B–D), `results.csv`, four dated session write-ups, and
`Pipeline3_v2_Final_Results_2026-07-28.md`. Where this record disagrees with an earlier document,
this record is the corrected version and says what it is correcting.
