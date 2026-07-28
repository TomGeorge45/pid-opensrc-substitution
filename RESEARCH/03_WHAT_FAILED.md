# What failed, and why

Grouped by kind of failure, each with the diagnosed cause rather than just the symptom. This is the
most useful document in the set: the negative results here are more transferable than the positive
ones.

---

## 1. The primary objective was never reached

**Stage-4 base selection never happened.** The plan was: bake off three candidates on Gupta, apply
the pass bar, lock one shared base. In practice one of three candidates (Molmo2) was benchmarked on
the real test set; Qwen3-VL and InternVL3 zero-shot runs were **deferred by decision, not blocked**,
and never run. The detection LoRA was never trained. All of Phase 6 — selection, pass-bar
application, base lock, reproducibility — was never started. **The master gate has zero boxes ticked
and `base.md` is still an empty template.**

The deeper reason is that the pass bar itself was unusable. The fallback-detector trigger was
specified as "a set fraction of Claude's Stage-4 recall, threshold to be set from the baseline run" —
**the fraction was never set.** And when the incumbent baseline finally ran, it scored F1 0.380
against a 0.70 bar, meaning the bar was calibrated against nothing real.

**The project drifted from its stated scope without updating its own charter.** `CLAUDE.md` says
"Stage 4 only" to the very end, while the majority of the last two weeks went into entity extraction
and relationship extraction. Directed and deliberate, but it means the repo's own top-level rules
document describes a different project than the one that was executed.

---

## 2. Fine-tuning failures — two full generations lost

**v1 and v2 domain-adaptation LoRAs were net-destructive.** Both used `all-linear` target modules,
which touches the vision tower. The result:

- v1 collapsed counting to "always answer 0" — **100% zero-answers against a 12% true zero-rate**
- v1 destroyed full-tile tag reading: **39% → 2%**
- v2 destroyed single-tag reading: **72% → 0%, on the exact crops it trained on**, while the
  untouched base scored 72%

Diagnosed cause: gradients that never reward glyph precision, rewriting the vision tower.

**Two separate warning signs were misread at the time.** Training loss collapsed to ~0.02 within 200
steps and stayed there. The first hypothesis was a masking bug — a model-free diagnostic reproduced
the exact masking logic and proved masking was **correct**. So the training was real; the model had
genuinely learned the templates and the dataset's statistical priors instead of visual grounding.
**Loss → 0 early is a warning, not a success signal.** And per-task loss curves expose this while a
blended average hides it — those curves were only added in v2.

**Templated targets are gameable.** One fixed phrasing per task lets a model drive loss to near-zero
without reading the image at all.

**Noisy pseudo-labels subtract capability.** The Tesseract "list all tags" task did not merely fail to
teach reading — it *unlearned* it. Qwen's untuned 39% agreement with Tesseract was an asset to
preserve, not a target to train against.

**Fixing it required abandoning the approach, not tuning it.** v3 excluded the vision tower entirely
and trained one adapter per stage. That worked (89.2% on two stages). The conclusion drawn was blunt:
general mixed-task training is destructive.

**Adapters then showed negative transfer across tasks.** On Stage 10.5 skid grouping:
- `v3-stage13` scored **24.5–61.5%**, below the 79.1% do-nothing baseline on 2 of 3 prompts
- `v2-general` and `v3-relation` scored 79.1% with **12/12 parse failures** — that number is
  fallback-to-baseline masking completely unparseable output, not a score

So a per-stage adapter is genuinely per-stage. It does not transfer, and it can make an adjacent task
worse than an untuned model.

---

## 3. The local relation validator does not work

This is the cleanest negative result in the repo, and it was established five different ways.

Qwen3-VL-8B asked a bounded yes/no question — "is there a line connecting these two boxed symbols?" —
scored **52.6% on 19 real crops, exactly the FALSE base rate.** Confirmed across five variants:
forced short answer, chain-of-thought at two token budgets, the fine-tuned adapter's own trained
prompt format, and adapter + CoT. Two variants degenerated into repeating `"No."` The best clean
result was 52.9%. **It never beat chance.**

In-pipeline it was worse: the validator **kept 0 of 8 candidates on all three sheets in both
configurations.** It does not filter, it deletes.

Its failure mode is specific and worth naming: **confident, plausible-sounding hallucinated traces**
that never actually check whether the traced path reaches the target. Not incoherence — fluent
wrongness.

**The adapter's own 89.2% benchmark did not transfer.** That number is real but was measured on
PID2Graph crops, the adapter's training domain. On real customer-PDF sheets it degenerated completely.
Compounding it: the adapter was only ever trained for **1 of 3 epochs** (paused at ~44,853/64,911
steps on a budget cutoff, never resumed), and its evaluation was **seed-disjoint, not file-disjoint**.

**No local model available cleared the bar.** GPT-5.5-low reached 83.3% only after correcting boxes
that had been drawn on tag text instead of symbols — and that is a frontier model, not a local one.

---

## 4. Measurement bugs that produced fake results

Four cases where the *metric* was wrong, not the model. Every one of them initially looked like a
model failure.

**A metric that could not score above zero.** The e2e relation F1 came back exactly 0.0 on all four
holdout sheets. Root cause: **PID2Graph contains zero direct equipment-to-equipment edges** — every
real connection routes through connector/crossing nodes — while the agent always emits direct
equipment-to-equipment relations. *No model could have scored above zero.* Fixed with a degree-4
contraction heuristic, validated against plausible degree distributions.

**A 90-point swing from a token limit.** An entire n=25 comparison table was an artifact:
`max_tokens=64` starved GPT-5.5's internal reasoning, returning empty strings that scored as
"undecided" (u20/25, u25/25, u13/25). Raising it to 2000 invalidated the whole table.

**A near-zero agreement score that measured the wrong category.** GPT-5.5's connectivity claims
agreed with traced geometry on only 3 of 31 pairs — while manual adjudication found the claims were
~80% correct. Cause: of ~25 true claims, **19–20 referenced off-page equipment that no single-sheet
tracer can ever confirm.** A measurement-category artifact, not an accuracy problem on either side.

**Parse failures masquerading as scores.** On the skid matrix, 12/12 unparseable outputs fell back to
a baseline and reported 79.1% — indistinguishable from a real result unless parse failures are counted
separately.

**Plus one self-inflicted case in the final session:** a first attempt at precision adjudication
marked as "real" exactly the claims already known to be real, producing a meaningless 89.5%. Caught
and discarded, but it is the same error class.

---

## 5. Approaches tested and refuted

**Symbol-extent resolution from vector geometry, unseeded — blocked.** `closePath` is False on all
5,608 tested paths even for genuinely closed shapes, and path bbox sizes span sub-pixel glyphs to
full-page rectangles. No global signal separates equipment outline from pipe from text stroke. Later
unblocked, not by a better algorithm but by a better input: a seed point inside the symbol.

**Off-page port location — two hypotheses refuted before the third worked.**
1. *The connector graphic encloses its text.* Refuted: 11 of 14 produced `radius_fallback` at
   confidence 0.00 — no enclosing path exists; the text has no box around it.
2. *Snap to the nearest loose pipe end.* Appeared to pass 22/22 within 0.2 in — **then the measurement
   was found to be circular.** The graph included the port text boxes, and the tracer masks symbol
   boxes, so it was measuring each box against its own mask boundary. 735 of 1,749 loose ends existed
   only because of those boxes. Unconfounded: median 183 px, only 8 of 22 within 108 px.
3. *The pentagon enclosing the sheet-number token.* Worked: 15/15, all at confidence 1.00. Found by
   rendering the border region and **looking at it** after two abstract hypotheses failed.

**Using a frontier model as a ground-truth source — proposed and rejected on data.** The idea was to
have GPT-5.5/Opus/Fable produce reference symbol coordinates. Refuted by measurement: **Molmo2 F1
0.6276 vs GPT-5.5-low 0.5125** on exactly that task. Worse, GPT-5.5's errors are *systematically* the
tag-text-vs-symbol confusion — the very variable under test — so it would have contaminated the
control group rather than merely adding noise.

**The distance cap.** Restored as the "missing half" of the production hop/distance caps. Swept at
four settings including disabled: known-truth **11/12 at every value**, while dropping **229 edges** on
one sheet. No measured benefit, measurable cost.

**The first child↔child suppression rule had inverted logic.** It keyed on whether two devices shared
a host equipment, keeping same-host pairs. But the measured false positives — two instruments hanging
off one vessel by separate stems — *share* a host. It caught 4 of 7. Replaced with a source-based test
(does one pipe run physically pass through both), which caught 6 of 7 with no real edge lost.

**Nemotron-Nano-VL-8B — ruled out with a diagnosed cause.** Given a symbol-grounding task it emits
DocVQA-style text-line bands: ~90 wall-to-wall horizontal strips, zero boxes on actual symbols.
Confirmed visually. Its training distribution, not a prompting problem.

---

## 6. Infrastructure and integration failures

**A silent monkeypatch that invalidated a whole run.** `ocr_reasoning.py` does
`from .vision import tiled_ocr_words`, binding the name into its own namespace — so patching
`pnid_pipeline.vision` never reached it. The first real run reported `route="agentic"` with
`call_llm.usage` **empty: the injected Qwen shim was never called at all.** The run looked like it
worked.

**A wrong default that invalidated an architecture premise.** The Phase 0 build log recorded
`PNID_MODE` unset as defaulting to `"cv"`. `git log -p` showed `mode: ocr_reasoning` since the file
was created. An entire section of the plan had been built around the wrong code path.

**A silent drop that forced a design decision.** A detection with an empty `value` produces **no
entity, no error, and no `unresolved` entry.** Confirmed by direct test: `value=None` → zero entities;
the same detection with `value="<a tag>"` → one entity. Consequence: **Molmo2 alone produces zero usable
entities**, because it emits points with no text. This is the entire reason the tag-matching design
exists.

**Stage gating that fails silently.** Multiple stages gate on
`page_classification == PID_DRAWING`. `stage_06_run` produces an empty `pages` list with no error if
it is unset — so a smoke test that skips stage 1 "for speed" looks like it passed while processing
zero pages.

**Environment cost.** Seven distinct Colab failures in one run alone: a Pillow 12.0.0 upstream
regression (whose obvious fix — uninstalling torchvision — was wrong, because Molmo2's
`trust_remote_code` file hard-depends on it), torchvision ABI pinning, PaddleOCR segfaulting natively
at every version tried, HF-hub version mixing. Plus a Colab trap: **`Restart session` restarts only
the kernel — pip changes persist on VM disk**, so a bad uninstall survives the restart.

**Two credential incidents.** VS Code's in-memory buffer re-saved a scrubbed HF token back into a
notebook config cell **twice** (verified never pushed). A live token was later found hardcoded in six
notebooks and removed at close.

---

## 7. Documentation and record-keeping failures

**Ground truth was lost.** The original 26-connection hand trace — the reference every relationship
recall figure depends on — was never written to disk. It survives only as a truncated prose summary,
from which 15 edges were reconstructed. **Permanently lost.**

**Numbers lived in prose, not code.** The three equipment extents and four instrument coordinates
behind the final F1 0.902 existed only in a chat transcript. The committed config file held a
*different* variant that reproduces a worse result. Fixed at close with a reproduction script — but
for most of the project the headline number was not regenerable.

**An architecture was cited to a document that does not describe it.** The four-arm union design is
referenced as living in `E2E_Harness_Plan.md`; that document contains only a two-arm design and the
tag-matching decision. The arms exist as executed notebooks and logged outputs, with the runner
scripts on HF rather than in the repo. **There is no plan document for the project's best-performing
local architecture.**

**A decision was cited repeatedly but never validated as written.** D-M1 specifies nearest-OCR-word
matching at radius = 1.5 × detection-bbox diagonal. `tag_matching.py` **does not exist**. A
differently-parameterised cousin (fixed 120 px radius) was built and run. So D-M1 was referenced
throughout as an established design while the thing actually validated was a variant of it.

**`results.csv` does not match its own prescribed schema.** `CLAUDE.md` specifies
`run_id,date,model,stage,ft_status,detection_mAP50,...`; the file has a different 8-column header, and
the e2e rows carry ~13 fields with their metrics buried in free-text notes. **No Stage-4 detection row
exists in it at all** — the headline detection numbers live only in a checklist markdown.

**Internal contradiction left unresolved.** `Stage4_Checklist_Status.md`'s per-phase tables show
Phases 0–3 largely complete while its own summary section says the same work is "not started" and the
project is "~70% through Phase 1." The tables are current; the summary is stale.
