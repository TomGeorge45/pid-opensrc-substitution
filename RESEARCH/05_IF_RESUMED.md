# State at close, and what a successor needs to know

The work is stopped. This document exists so that anyone picking it up does not re-walk ground already
covered, and does not trust numbers further than their evidence allows.

---

## 1. What actually works and can be relied on

| Component | State | Evidence |
|---|---|---|
| Conversion layer (`src/e2e_bench/`) | **Working, verified** | 25/25 tests pass, including a full 8-stage chain against real agent code on a real P&ID image |
| E2E harness driving the real agent | **Working** | 4-sheet holdout runs completed, zero blockers, real stage 4/6/12/13 code |
| Multi-arm union entity extraction | **Working, best local result** | Beats best single arm on 4/4 sheets, up to 0.909 revR |
| Molmo2 pointing LoRA v1 | **Working** | F1 0.732 on the 20-sheet Gupta gate — best detection number produced |
| v3-stage13 adapter | **Working** | 89.2% vs GPT-5.5's 66.7%, n=120, file-disjoint |
| Relationship pipeline, deterministic half | **Working on its dev sheet** | F1 0.902, all 42 claims adjudicated, reproducible via script |
| Off-page connector (port) detection | **Working** | 15/15 pentagons at confidence 1.00; boundary edges 16/16 correct |
| Agent facts (tiling, schema, ontology) | **Code-verified** | `Agent_Pipeline_Facts.md`, read from source |

## 2. What does not work, and should not be retried as-is

- **The local relation validator.** 52.6% across five prompt variants, 0/8 kept in-pipeline. Do not
  spend more GPU time on it without a genuinely held-out customer-PDF-domain eval and the remaining
  training epochs finished.
- **General mixed-task fine-tuning.** Two generations, both destructive. Language-only, per-stage only.
- **Nemotron for detection.** Emits DocVQA text bands, diagnosed root cause.
- **ModelScope as a download path.** 200× slower than HF, and its threads cannot be killed.
- **Google Drive.** Structurally incompatible with unattended runs.
- **The distance cap in relation building.** No measured benefit at any setting, 229 edges dropped.

## 3. The highest-leverage unexecuted levers

Ordered by expected return per unit of effort. All were identified during the project and none were done.

1. **Port the real 916-line production prompt to the local models.** Zero GPU cost, zero training.
   Every local zero-shot number in this repo was measured with simpler custom prompts, and prompt
   changes alone moved skid grouping from 47.6% to 92.3% and title-block extraction from 0.33 to 0.67.
   This is the single cheapest untested improvement.
2. **Build the recheck pass and failed-tile retry sweep.** Pure harness engineering, no model change.
   Two of the five mechanisms behind the incumbent's resilience, both absent locally.
3. **Finish v3-relation's remaining ~2 epochs.** It stopped at 1 of 3 on a budget cutoff and still
   cleared its adoption bar.
4. **Run the deferred Qwen3-VL and InternVL3 Stage-4 zero-shot benchmarks.** The three-way comparison
   the project was built around covers 1 of 3 candidates. InternVL3 is neither ruled in nor out.
5. **Produce the Arm P vs Arm L cascade comparison.** The e2e plan's stated deliverable. Only Arm P has
   cascade numbers; there is no Arm L row.
6. **Retrain v3-relation against a frozen sheet-level PID2Graph holdout.** Named "the methodologically
   correct fix" and deferred. Until then, any Arm-L-beats-Arm-P relation result is weaker than it looks.
7. **Wire the multi-arm union into the relationship stage.** Relationship tests used prod's OCR/Vision
   tag extraction, which carries text-label positions but not symbol shapes — the recurring bug source.
   Molmo2's points would fix it at source, and its pointing works on rasters as well as PDFs.

## 4. Numbers not to quote without re-deriving

- **The Molmo2 "improved config beats the incumbent" claim** and the **GPT-5.5 "best zero-shot
  detector" claim.** Neither was ever re-verified; the checklist itself says to pull them fresh.
- **Anything at n≤25.** Includes the entire early GPT-5.5 comparison table, which was an artifact of a
  64-token limit starving the model's reasoning.
- **Stage 10.5 and Stage 2 results** — n=12 and n=6 respectively, and Stage 10.5's ground truth is
  script-constructed with a dashed boundary drawn onto the image, not real.
- **Relationship recall (86.7%)** — measured against a 15-edge partial reconstruction of a lost
  26-connection trace, model-traced rather than human-certified.
- **Relationship precision (93.9%)** — self-adjudicated by the same party that wrote the pipeline, and
  6 of 42 claims remain unadjudicated. Bounds are [73.8%, 95.2%].
- **The "0.140 → 0.780" union improvement** — two different runs with different configs, not one
  before/after experiment.
- **The Phase 0 legacy regression gate (78/52)** — no longer reproduces. Legacy now returns 221/95
  because the tracer populates `passes_through_symbols`, so Pass 0 fires there for the first time.
  Expected consequence of a real fix, but it means refactor-neutrality can no longer be claimed.

## 5. Traps that will otherwise be re-hit

- **`from .module import name` defeats monkeypatching.** Patch the importing module's namespace, not the
  source module. A run that reports a route with empty `call_llm.usage` never called your model.
- **A detection with empty `value` is silently dropped** — no entity, no error, no unresolved entry.
- **Multiple stages gate on `page_classification == PID_DRAWING`.** Skipping stage 1 "for speed" makes
  `stage_06_run` process zero pages while appearing to pass.
- **`confidence` is at `provenance.confidence`; the symbol bbox is at `provenance.bbox`** (absolute
  xyxy, OCR-anchor-corrected). `TitleBlockRecord.bbox_drawing` is xywh while everything else is xyxy.
- **Colab's `Restart session` restarts only the kernel** — pip changes persist on VM disk.
- **Molmo2 needs `transformers==4.57.1`** while the others run 5.12.1. Pin `pillow<12` and keep
  torchvision (Molmo2's remote code hard-depends on it).
- **A second Molmo2 load OOMs** even on an 80 GB A100 with Qwen co-resident.
- **HF checkpoint pushes accumulate LFS blobs.** 211 commits consumed 82.5 GB of a 100 GB tier.
- **The entity ontology is per-tenant at runtime**, not a fixed enum. Any eval assuming a fixed class
  list is testing something the agent does not do — and "coverage %" has no honest denominator.

## 6. Reproducing what exists

```
# the relationship-stage headline result
PYTHONPATH=src/relation_bench .venv-e2e/bin/python \
  scripts/reproduce_sheet1_final.py <sheet-stem> <path-to-that-sheet.pdf>
# expect: precision 93.9%, recall 86.7%, F1 0.902, 42 claims, 0 violations
```

Durable artifacts on Hugging Face (`timthy45/pnid-extraction-datasets`): source sheets
(the two source-sheet archives under `sheets/`), the PID2Graph corpus
(`pid2graph/PID2Graph.zip`, 9.3 GB), extraction outputs (`benchmarks/extraction_2026-07-24/`), probe
and candidate-crop bundles, and the 42-crop review bundle. Model checkpoints live at
`timthy45/qwen3vl-pnid-domain-base` (`v3-stage13/latest`, `v3-relation/latest`, `v2/latest`) and
`timthy45/molmo2-pnid-pointing-lora/v1`.

In-repo: `benchmarks/sheet1_adjudication_42claims_2026-07-27.json` (32 real / 7 false / 3 unsure with
per-claim reasoning), `src/relation_bench/hand_extents/sheet1.json` (both extent variants, the four
recovered seeds, the 15-edge GT), and `benchmarks/pipeline3_artifact_source_2026-07-28.html`.

## 7. Two housekeeping items

**Rotate the Hugging Face token.** One was hardcoded in six notebooks and removed at close. It was
never committed, so git history is clean — but it sat in plaintext on disk across multiple files and
there were two prior near-miss incidents where an editor re-saved a scrubbed token.

**This repository is public and contains real customer drawing content.** The PDFs are gitignored, but
sheet identifiers and equipment tags (`VESSEL-1`, `SHEET-1`, `TREATER-1` and others) appear
throughout the documentation and results. If any of that is customer-identifying, it is far cheaper to
address now than after the repo has been cloned.
