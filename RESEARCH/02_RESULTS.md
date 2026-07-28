# Results — every measured number

Organised by stage. Each figure carries its sample size and reliability tier. Where an earlier
document in this repo states a number more confidently than its evidence supports, the correction is
noted inline.

---

## 1. Data integrity (Phase 1, all confirmed)

- **Gupta:** 92 sheets = 72 train + 20 test, asserted. 0 orphans, 0 unannotated. Every box labelled
  "Symbol" — **class-agnostic, no type information at all.** This is why the two-part metric exists.
- **Kaggle:** the original copy was **broken — only 6,591 of 30,000 labels present**. Re-downloaded:
  30,000/30,000 labelled, **195,759 instances, 32 classes**, matching the dataset card. An earlier
  43,055-instance figure is superseded — it was measured on the broken copy.
- **Kaggle ontology coverage:** 6/6 illustrative categories (100%) by category count, but
  instance-weighted the distribution is heavily skewed — valve 48.2%, pipeline 17.4%, measurement
  17.0%, nozzle 9.1%, asset 5.4%, safety_device 2.9%. **The skew is the number to report, not the
  flat 100%.**
- **Tiling** implemented to the agent's exact spec (1024 px tiles / 205 px overlap / 819 px stride,
  edge tiles clamped) and visually verified. Known simplification: no title-block carve-out.

---

## 2. Stage 4 — symbol detection

### The headline finding

| Model | Precision | Recall | F1 | Parse failures | Tier |
|---|---|---|---|---|---|
| Molmo2-O-7B zero-shot, baseline config | 0.538 | 0.364 | **0.434** | 10.2% (13/127) | Measured |
| **claude-sonnet-4-6 (the incumbent being replaced)** | 0.392 | 0.369 | **0.380** | 0/127 | Measured |

20 frozen Gupta test sheets, 127 tiles. Provisional pass bar: **F1 ≥ 0.70**.

**Neither clears it.** The model this project set out to replace scores 0.380 against a 0.70 bar on
real detection. That single fact reframes the whole effort: the target was not a high one, and the
gap being closed was smaller than assumed.

**Molmo2's weakness is density, measured not hypothesised.** Sheets needing 1–2 tiles score F1
0.88–1.00 (three sheets at exactly 1.00). Sheets needing 12–20 tiles with 100+ ground-truth symbols
collapse to F1 0.13–0.52 — worst case sheet `233`, 199 GT symbols, **F1 0.13**. Recall is the primary
failure mode.

### Parse-rate screening (n=10, Kaggle — deliberately the wrong dataset for detection scoring)

| Model | Parse success | Notes |
|---|---|---|
| Molmo2-O-7B | **10/10**, avg 4.40 s/tile | Best on parse and speed. `confidence` and `entity_type` are `None` on every detection — no native fields |
| InternVL3-8B | 6/10 | Failure not cleanly density-driven |
| Qwen3-VL-8B | 4/10 | Below the 95% bar. **Every tile with ≥9 GT boxes times out at exactly 60 s**; tiles with ≤5 succeed in <11 s. One hallucinated detection at confidence 0.98 **on blank canvas** |

### Later Stage-4 numbers — and an honest gap

A **Molmo2 improved config** (512 px tiles, 2× upscale, autocontrast) reportedly beat the incumbent
on the full 20 sheets, and **GPT-5.5-low** was recorded at one point as the best zero-shot detector.
**Neither figure is certified** — the checklist itself says to pull them fresh rather than quote from
memory, and no side-by-side re-check was ever run.

The one Stage-4 figure that *is* clean, from a later frozen-test-set run:

| Model | Precision | Recall | F1 | Tier |
|---|---|---|---|---|
| **Molmo2-points** (512 px, 2× upscale, enhance) | 0.6309 | 0.6244 | **0.6276** | Measured |
| **GPT-5.5-low** (`v3-scan-fenced`, 1024 px) | 0.5335 | 0.4932 | **0.5125** | Measured |

**Molmo2 beats the frontier model at symbol localisation by ~22% relative.** This is one of the
project's most useful results and it is counter-intuitive: Molmo2 has a native pointing head, while
GPT-5.5 emits coordinates as text tokens. It also means a frontier model cannot serve as a
ground-truth source for symbol positions — a proposal that was raised and rejected on this evidence.

### What was never done

Phase 4's Qwen3-VL and InternVL3 zero-shot runs on the real test set were **deferred by decision**,
so the three-way comparison covers 1 of 3 candidates. The detection LoRA (5.3), the fine-tuned test
run (5.4), and **all of Phase 6 — selection, pass-bar application, base lock, reproducibility** —
were never started. **The Stage-4 master gate has zero boxes ticked, and `base.md` remains an empty
template.**

---

## 3. Domain-adaptation LoRA — three generations

### v1 and v2 (mixed-task, `all-linear` LoRA, touches the vision tower)

| Task | Base | v1 | v2 |
|---|---|---|---|
| Relation accuracy | 56–77% | **90%** | 88% |
| Typed-summary class F1 | 0.00 | **0.36** | 0.34 |
| Count: answered "0" (truth 12%) | 6% | **100%** ✗ | 0% ✓ |
| Count MAE | 16.3 | 15.0 | **12.3** |
| Tag reading (single-tag crops) | **72%** | — | **0%** ✗ |
| Full-tile tag listing | **39%** | **2%** ✗ | 2% ✗ |

Two rounds, two task designs, the same outcome: **fine-tuning destroyed reading ability.** v2 scored
**0% on the exact crops it trained on** while the untouched base scored 72%. Diagnosed cause:
all-linear LoRA rewriting the vision tower with gradients that never reward glyph precision.

### v3 — per-stage, language-only LoRA (vision tower excluded)

| Task | GPT-5.5-low | Qwen base | v3 adapter | Old v2 general |
|---|---|---|---|---|
| Entity validation (stage 13) | 66.7% | 45.0% (below chance) | **89.2%** | 35.0% (below chance) |
| Relation validation (stage 12) | 72.5% | 80.0% | **89.2%** | 84.0% (noisy) |
| Text extraction (stage 5) | 98.5% | **100.0%** | not built | 66.7% |

n=120 for entity/relation, n=65 for text extraction. Excluding the vision tower fixed the
destruction problem outright.

**Two caveats that matter.** v3-relation is **only partially trained** — paused mid-epoch-0 at
~44,853 of 64,911 steps on an 8-hour budget cutoff, never resumed. And its evaluation is
**seed-disjoint, not file-disjoint** — same source trees for train and eval. Not fully rigorous.

**Stage 5 got no adapter by decision:** the untouched base already wins at 100%, and every fine-tune
attempt made it worse.

---

## 4. Other stages

### Stage 2 — title block (n=6, directional only)

| Model | All-4-fields correct |
|---|---|
| GPT-5.5-low | 0.67 |
| Qwen3-VL-8B, prompt v2 | 0.67 |

A tie, with complementary field strengths. Qwen's **first generic prompt scored 0.33** — prompt
tuning closed the entire gap. n=6 because only ~10% of Gupta sheets are real titled drawings.

### Stage 10.5 — skid grouping (n=12, constructed ground truth)

No real ground truth exists, so GT was constructed by script and a dashed boundary drawn onto the
image. **The trivial "everyone separate" baseline scores 79.1%** on this distribution — the number
every result must be read against.

| Run | Score |
|---|---|
| Qwen base × v3-per-symbol prompt | **92.3%** — edges out GPT-5.5-low |
| GPT-5.5-low | 91.9% |
| Qwen base × v2-anti-merge prompt | 56.4% — **below the do-nothing baseline** |
| Qwen base × v1-shared prompt | 47.6% — **below baseline** |
| Qwen + v3-stage13 adapter | 24.5–61.5% — **below baseline on 2 of 3 prompts** |
| Qwen + v2-general / v3-relation | 79.1% with **12/12 parse failures** — the score is fallback masking total garbage, not a result |

Two findings here. **Prompt design dominated model choice** — the same base swung from 47.6% to
92.3% on prompt alone. And **adapters showed real negative transfer**: an adapter trained for entity
validation made a different task actively worse than doing nothing.

GPT-5.5's 91.9% also failed a metamorphic self-consistency check: with the same boundaries and
shuffled labels, only **4 of 12 crops** were fully self-consistent.

### End-to-end cascade through the real agent (Arm P = GPT-5.5-low)

4-sheet holdout, running the real stage 1/4/6/12/13 code:

| Sheet | Entity F1 pre→post stage-13 | Relation F1 pre→post stage-12 |
|---|---|---|
| OPEN100/8 | 0.414 → **0.310** (stage 13 removed correct entities) | 0.0 → 0.0 |
| OPEN100/1 | 0.274 → 0.323 | 0.0 → 0.0 |
| DatasetPID/246 | 0.685 → 0.719 | 0.077 → 0.108 |
| DatasetPID/443 | 0.682 → 0.692 | 0.173 → 0.185 |

**Stage 13 helped 3 sheets and hurt 1** — not a uniform net positive. Entity F1 is much higher on
synthetic Dataset PID (~0.7) than real OPEN100 (~0.3), unexplained.

**A methodology bug worth recording.** The first pass returned relation F1 exactly 0.0 on all four
sheets. Root cause: **PID2Graph contains zero direct equipment-to-equipment edges** — every real
connection routes through connector/crossing nodes — while the agent always emits direct
equipment-to-equipment relations. *The metric could not score above zero by construction, regardless
of model quality.* Fixed with a degree-4 contraction heuristic, validated by comparing degree
distributions across three variants (naive tunnelling gave mean degree ~11, implausible;
block-everything gave 0.27, mostly isolated; degree-4 gave ~1.9–2.1, plausible).

### The cloud baselines being measured against (14 sheets, `revR`)

| Model | mean revR | $/drawing | sec/drawing |
|---|---|---|---|
| GPT-5.5 high | **0.836** | 4.21 | 982 |
| GPT-5.5 low | 0.813 | 0.96 | 295 |
| Sonnet 4.6 | 0.811 | 0.62 | 201 |
| Gemini 3.1 Pro | 0.752 | 0.78 | 227 |

Proposed bars: ≥0.75 credible POC, ≥0.813 parity headline. Note the cost spread — GPT-5.5-high is
**4.4× the price and 3.3× the latency** of low for +0.023 revR.

### The multi-arm union — the best local result the project produced

Four arms, unioned by tag text: **Arm 0** deterministic regex over OCR words (no model), **Arm 1**
whole-page single-pass Qwen3-VL, **Arm 2** Molmo2 pointing → Qwen reads each crop, **Arm 3** CV-hybrid
through the agent's `read_shapes`/`read_regions` path.

| Sheet | Arm 0 | Arm 1 | Arm 2 | Arm 3 | **Union** |
|---|---|---|---|---|---|
| GD-B-540-DP-2920-005 | 0.028 | 0.055 | 0.606 | 0.550 | **0.835** |
| GD-T-435-DT-2042-056 | 0.000 | 0.818 | 0.182 | 0.636 | **0.909** |
| PX-2365-0140006-001 | 0.149 | 0.040 | 0.524 | 0.669 | **0.843** |
| PX-2368-0180004-001 | 0.200 | 0.120 | 0.460 | 0.540 | **0.720** |

**The union beats the best single arm on all four sheets**, and no single arm is consistently best —
Arm 1 wins the small sheet at 0.818 and is nearly worthless on the big ones at 0.040–0.055. On an
earlier 3-arm run the union reached **0.850 on PX-2365-0150022-001, beating GPT-5.5's 0.75 on that
sheet.** Directional (n=4 and n=2 sheets, no aggregate mean was ever computed), but it is the closest
this project came to frontier parity with local models.

Arm 2's pointing adapter is worth recording separately: **`molmo2-pnid-pointing-lora/v1` scored
F1 0.732** on the 20-sheet Gupta gate — the best Stage-4 detection number the project produced,
better than the 0.6276 zero-shot config. **A v2 attempt regressed to 0.686** and was not used.

**Two corrections to claims made elsewhere in this repo.** First, the widely-cited "0.140 → 0.780"
improvement is **not a single before/after experiment** — 0.140 is single-arm `L-ocr` Qwen and 0.780
is a zero-shot 6-prompt union, from different runs with different configs. Both are real logged
numbers; the framing was wrong. Second, `revR` is arguably the wrong ruler entirely: it scores tag
*text* recall, and none of the datasets carrying tag-text truth also carry position data.

### Local entity extraction — Qwen vs GPT-5.5 (3 dev sheets, revR)

| Sheet | Qwen3-VL-8B | GPT-5.5-high |
|---|---|---|
| GD-T-435-DR-2031-030 | **0.810** | 0.841 — **96% of GPT, near-parity** |
| PX-2365-0150022-001 | 0.183 | 0.75 |
| PX-2368-0180004-001 | 0.140 | 0.98 |
| Mean | **0.378** | 0.857 |

**A family split, diagnosed rather than guessed.** GD sheets have simple regular tags and Qwen nearly
matches the frontier model. PX truth is dominated by multi-token compound line-tags
(`10"-WS-2630-48-MSDX1-HSW`) and suffixed bubbles (`LAL2160A`) where one dropped trailing character
scores zero. **A read-precision gap, not an extraction-strategy gap.**

Confounds recorded honestly: GPT-5.5's numbers used high reasoning effort (~980 s and $4.20 per
sheet, versus Qwen's ~470–700 s under a 4k-token cap) with a prompt co-tuned over 11 recorded
iterations against this exact corpus.

Prompt iteration history, each round fixing an observed failure: round 1 (JSON-Schema dump) → Qwen
echoed schema defaults, **zero tags on every sheet**; round 2 (concrete example) → GD reached 0.810
but one sheet **dumped OCR token IDs as tag texts**; round 3 (forbid bare numbers, enumerate tag
categories) → fixed the ID dump (+0.033) and regressed another sheet.

---

## 5. Relationship extraction — the deepest-measured stage

### Probes that gated the design

| Probe | Result | Verdict |
|---|---|---|
| Symbol-extent reconstruction from vector geometry, seeded | Passed incl. branching case | PASS |
| Qwen reading off-page connector text (n=8) | 87.5% | PASS |
| **Qwen bounded connectivity verification (n=19)** | **52.6%** — exactly the FALSE base rate | **FAIL** |
| GPT-5.5-low, same task, original boxes | 68.4–73.7% | Below bar |
| GPT-5.5-low, same task, **corrected boxes** | **83.3%** | Clears bar |

Qwen's 52.6% was confirmed across **five** prompt variants — forced short answer, two chain-of-thought
settings, the adapter's own trained format, adapter+CoT. It never beat chance. Two variants collapsed
into repeating `"No."`. Its failure mode is confident, plausible-sounding hallucinated traces.

**The box-quality finding is the important one:** GPT-5.5 moved from 76.5% to **83.3%** purely by
correcting three boxes that had been drawn on tag *text* instead of the drawn *symbol*. Measurement
error was hiding real capability.

### The in-pipeline validator result

The local relation validator (Qwen + v3-relation) **kept 0 of 8 candidates on all 3 sheets in both
configurations.** It does not filter — it deletes. This reproduced Probe 2 inside the real pipeline
stage, and is why every relationship number quoted is a *pre-LLM* deterministic result.

### Deterministic pipeline, final state (one sheet, hand-assisted)

| Metric | Before v2 | After v2 | Final |
|---|---|---|---|
| Precision | — | 82.1% | **93.9%** |
| Recall (15-edge GT) | 58% strict / 77% near-miss | 80.0% | **86.7%** |
| F1 | — | 0.810 | **0.902** |
| Boundary (off-page) edges | 0 | 16/16 | **16/16 = 100%** |
| Isolated symbols | 13/38 | — | 9/34 |

All 42 claims were individually adjudicated against the drawing: 32 real, 7 false, 3 unsure. The
final rule change removed 6 of the 7 false positives with **no adjudicated-real edge lost**.

**Boundary detection is the standout** — off-page connections went from *structurally impossible to
verify* to 16/16 correct, by locating ports at the pentagon enclosing each sheet-reference token
(15/15 resolved, confidence 1.00, consistently 203×42 px).

### Multi-sheet reality check

Against PID2Graph — the only multi-sheet corpus with real edge ground truth — the pipeline scores
**F1 0.225 versus a trivial nearest-neighbour floor of 0.226.** That is the raster path on a corpus
with a documented tracing ceiling (R1 alone scores 0.023, *below* the floor, root-caused to a
corpus-rendering mismatch that threshold sweeps and extra masking both made worse). Not comparable to
the vector numbers, but it is the only multi-sheet figure that exists.

**And the vector result does not generalise.** On the two other development sheets, extent resolution
drops from 86% to **19% and 43%**, and port detection fails entirely on the sheet whose CAD export
outlines its text into vector paths (1 extractable word on the whole page).

---

## 6. The dataset world-scan

A deliberate search for usable P&ID training data. The negatives are the valuable part.

**Licensing traps caught:**
- **Digitize-PID / Dataset-P&ID (Paliwal 2021): CC BY-NC-ND 4.0 — blocked for commercial use.**
  This is the 500-sheet synthetic component of the corpus the project had been using.
- **FUNSD and XFUND: confirmed non-commercial.**
- **CGHD:** GitHub badge says CC0-1.0; the authoritative Zenodo record says CC-BY-4.0.

**Confirmed dead ends — stop searching these:**
- ISA and ASME publish no open datasets, paid standards only.
- IEEE DataPort has zero P&ID datasets.
- No university or lab-hosted P&ID corpora beyond those already covered.
- No synthetic generators beyond Digitize-PID, pid_reader (GPL-3.0), SynthPID.
- **Stage 13 has no public dataset anywhere** with keep/correct/remove judgments on engineering
  drawings. It must be purpose-built — a verified conclusion, not an assumption.

**Corpus exhaustion:** the local PID2Graph mirror *is* the complete public release — 12 real OPEN100
sheets plus 500 synthetic. "OPEN100" is not a sheet count. There is no larger version to fetch, and
both trees were fully consumed by v3-relation's training sampling.
