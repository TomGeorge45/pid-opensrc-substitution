# Takeaways

The transferable content. Ordered roughly by how much they cost to learn here.

---

## On measurement

**1. Verify the metric can express the answer before trusting a zero.** The e2e relation F1 returned
exactly 0.0 across four sheets and looked like total model failure. The real cause was that PID2Graph
has no direct equipment-to-equipment edges while the agent only emits them — *no model could have
scored above zero.* A metric that cannot produce a positive result is indistinguishable from a model
that deserves none. Sanity-check by asking: what input would make this metric return 1.0?

**2. Recall alone will mislead you, and the project's own rule was still nearly violated.** The
relationship stage was reported at "58–77% recall" for days before anyone asked what fraction of its
*claims* were real. When precision was finally measured, the pipeline was emitting 42 claims to cover
15 known connections. Precision is the cheaper half to measure (review what you claimed) and recall is
the expensive half (find what you missed) — which is exactly why precision gets skipped.

**3. Count parse failures separately, always.** 12/12 unparseable outputs fell back to a baseline and
reported 79.1% — a plausible-looking score representing total garbage. Any fallback must be visible in
the number or reported beside it.

**4. Small n reverses.** A documented case: an adapter "beat" the base 84% to 80% at n=25; one flipped
answer erased the entire margin, and at n=120 the picture changed. The project settled on n≥100 for
any claim it intended to act on. Below that, say "directional" and mean it.

**5. Always have a do-nothing baseline.** The trivial "everyone separate" baseline scored **79.1%** on
skid grouping. Two configurations that looked like results — 47.6% and 56.4% — were in fact *worse
than doing nothing*, which is invisible without the floor. Similarly, nearest-neighbour connection
scores F1 0.226 on PID2Graph, and the full pipeline scored 0.225.

**6. Do not let the party being measured author the ground truth.** Attempted twice here. Once by
proposing a frontier model as a coordinate reference — rejected because its errors were *systematically*
the variable under test, so it would have contaminated the control rather than adding noise. Once by
self-adjudicating precision on the subset already known to be real, producing a meaningless 89.5%.

**7. Loss → 0 quickly is a warning.** Two training generations were lost to it. Templated targets let
a model minimise loss without reading the image. Per-task loss curves expose this; a blended average
hides it.

---

## On fine-tuning vision-language models

**8. Excluding the vision tower is not an optimisation, it is the difference between working and
destructive.** `all-linear` LoRA took single-tag reading from 72% to **0% on the exact crops it trained
on**. Language-only LoRA on the same base reached 89.2% on two stages. Same data, same model, opposite
outcome.

**9. Noisy pseudo-labels can subtract capability.** Training on Tesseract's "list every tag" output
didn't fail to teach reading — it *unlearned* it, 39% → 2%. An untuned model's existing ability is an
asset with a value; check you are not training over something better than your labels.

**10. Per-stage adapters are per-stage, and can be actively harmful elsewhere.** An entity-validation
adapter scored *below the do-nothing baseline* on skid grouping. Two others produced 100% unparseable
output on a task they weren't trained for. Budget for one adapter per job, and re-test every adapter on
every stage it might touch.

**11. Domain adaptation is a foundation, not a product.** No 7K-example LoRA competes with a frontier
model at general reasoning, and it doesn't need to — it needs to win at a handful of narrow, repetitive
jobs. The project's framing of this was right even where execution fell short.

---

## On the substitution premise itself

**12. Measure the incumbent before assuming the gap.** The model this project existed to replace
scored **F1 0.380** on real symbol detection against its own 0.70 bar. Knowing that on day one would
have reframed everything — the goal was never "match an excellent system," it was "match a mediocre
one," which is a different and much more achievable project.

**13. A frontier model is not uniformly better.** **Molmo2 beat GPT-5.5-low at symbol localisation,
F1 0.6276 vs 0.5125** — a ~22% relative win for a 7B open model. Molmo2 has a native pointing head;
frontier models emit coordinates as text tokens. Pick per capability, not per reputation.

**14. Prompt design can dominate model choice.** On skid grouping the same base swung from **47.6% to
92.3%** on prompt alone — from below-baseline to beating a frontier model. On title-block extraction a
generic prompt scored 0.33 and a tuned one 0.67, closing the entire gap to GPT-5.5. And the project's
own highest-leverage identified item — porting the production agent's real 916-line engineered prompt
to the local models — **was never done.** Every local zero-shot number here was measured with simpler
custom prompts.

**15. Ensembling cheap diverse arms beat making one arm good.** The four-arm union beat the best single
arm on **all four** sheets (up to 0.909), and no single arm was consistently best — one arm scored 0.818
on a small sheet and 0.040–0.055 on large ones. Union reached 0.850 against GPT-5.5's 0.75 on one sheet.
This was the closest the project came to parity, and it came from combining weak signals rather than
perfecting a strong one.

**16. Pipeline resilience substitutes for model accuracy.** The incumbent's advantage was analysed as
tiling + low-confidence recheck + failed-tile retry + downstream validation, letting mediocre per-call
accuracy survive. Two of those mechanisms — the recheck pass and retry sweep — were **never built
locally**, and they are pure harness engineering with no model cost.

---

## On data

**17. Check licences before building on a corpus.** **Digitize-PID / Dataset-P&ID is CC BY-NC-ND —
blocked for commercial use** — and it is the 500-sheet synthetic half of the corpus this project
trained a relation adapter on. FUNSD and XFUND are likewise non-commercial. CGHD's GitHub badge
(CC0-1.0) contradicts its authoritative Zenodo record (CC-BY-4.0). For a project whose entire purpose
is a commercial substitution, this is a first-day check, not a later one.

**18. Verify a downloaded dataset against its own card.** The Kaggle copy in use had **6,591 of 30,000
labels**. Everything measured on it was wrong, and an early instance count had to be retracted.

**19. Absence of data is a finding worth recording.** ISA and ASME publish nothing open; IEEE DataPort
has zero P&ID datasets; no university corpora exist beyond those already found; **Stage 13 has no
public dataset anywhere** and must be purpose-built. These negatives stop the next person repeating a
multi-day search.

**20. Corpus exhaustion is real and easy to miss.** Sampling walked 98.2% of one tree's patches and
20.6% of the other, but computed expected untouched sheets at **≈0.0 of 12 and ≈0.065 of 500** — i.e.
training almost certainly touched every sheet in both. A seed-disjoint eval split is not a
file-disjoint one, and the difference invalidates exactly the claim you want to make.

---

## On working with a production codebase

**21. Drive the real code; don't reimplement it.** Every harness here runs the actual agent's stage
functions through client shims mimicking `AsyncAnthropic.messages.create()`. That is why the numbers
describe the real pipeline. A ~60-line fake client was enough, and two planned reimplementations
turned out never to be needed.

**22. Read the serialized schema, not the prose or the internal class.** Three flags, each a silent
failure waiting: `confidence` lives at `provenance.confidence` (the internal class has it top-level and
is never serialized); the symbol bbox is `provenance.bbox`, absolute xyxy, and **OCR-anchor-corrected**
so it is neither the raw model box nor a union of OCR words; `TitleBlockRecord.bbox_drawing` is xywh
while everything else is xyxy.

**23. Silent drops are worse than errors.** A detection with an empty `value` yields no entity, no
error, no unresolved entry. That single behaviour meant **Molmo2 alone produced zero usable entities**
and forced an entire design decision.

**24. `from .module import name` defeats monkeypatching.** Patching `pnid_pipeline.vision` never
reached `ocr_reasoning`'s already-bound local name. The run completed, reported a route, and had
**never called the injected model at all.**

**25. Verify defaults in git history, not documentation.** A build log recorded `PNID_MODE` defaulting
to `"cv"`; `git log -p` showed `ocr_reasoning` since file creation. A section of the plan was built on
the wrong code path.

---

## On the meta-problem: continuity across sessions

The project ran as three sequential AI sessions with handover documents. That worked — but the failure
modes are specific and worth designing against.

**26. If a number is not in a file, it does not exist.** The 26-connection ground truth was never
written to disk and is permanently lost. The hand-placed coordinates behind the final headline figure
lived only in a transcript, and the committed config held a *different* variant that reproduces a worse
result. **Persist the exact input that produced any number you intend to quote, plus a script that
regenerates it.**

**27. Treat ephemeral working directories as hostile.** Scripts, renders and reference data were left in
session-scoped temp space. Some was rescued; some was already gone.

**28. Verify inherited handovers against reality before building on them.** The third session checked
its handover against `git status` and found three material errors — including files declared
unrecoverable that were still present, and a completed upload flagged as unverified.

**29. Documents drift; add corrections rather than trusting the latest prose.** Several confident claims
in this repo needed retracting — a "0 of 8,105" measurement that was correct when taken but wrong once a
fix landed; a "0.140 → 0.780" improvement that was two unrelated runs; an architecture cited to a
document that never described it; a decision cited as validated whose implementation does not exist.
None were dishonest. All were prose outliving its evidence.

**30. Two sessions on one task will both do it.** The same prompt reached two sessions once and produced
two implementations of one fix plus two non-matching reruns of the same benchmark. Neither was wrong;
they measured different inputs. Cheap to avoid, confusing to unpick afterwards.
