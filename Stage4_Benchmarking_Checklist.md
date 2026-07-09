# Stage 4 — Symbol Detection Benchmarking · Task Checklist


**Revision:** 3.0 — precision in pass bar (mAP/F1 not recall), real-typing limitation logged, rare-class + coverage + Part-B-mode slips fixed
**Goal:** Pick the base VLM (Qwen3-VL vs InternVL3 vs Molmo) by benchmarking symbol detection against Gupta ground truth, then domain-fine-tune the winner and train a detection LoRA. This decision unlocks every downstream stage.

**How to use:** every task has a paired **✓ Confirmation** — an explicit check that must pass before moving on. Do not proceed to the next task until its confirmation passes. Confirmations are concrete (a count, a file, a number), never "looks fine."

**Legend:** ☐ not started · ◐ in progress · ☑ done + confirmed

---

## Phase 0 — Environment Setup

### 0.1 Provision Colab Pro+ session
- ☐ **Task:** Start a Colab Pro+ runtime; select GPU.
- ☐ **✓ Confirm:** `!nvidia-smi` prints a GPU (A100 40 GB ideal; L4/T4 acceptable). Record which GPU and its VRAM. If VRAM < 24 GB, flag it — 8B fine-tuning may need QLoRA/offload.

### 0.2 Mount Drive for persistence
- ☐ **Task:** Mount Google Drive; create project folder `MyDrive/pid_stage4/`.
- ☐ **✓ Confirm:** `os.path.exists('/content/drive/MyDrive/pid_stage4')` returns `True`. Write a test file, read it back, confirm contents match.

### 0.3 Install dependencies
- ☐ **Task:** Install torch, transformers, vllm/lmdeploy, mlflow, pycocotools, supervision (or equivalent detection-metric lib), kagglehub, and the model-specific loaders.
- ☐ **✓ Confirm:** every package `import`s without error in a fresh cell; print each version. Pin versions to a `requirements.txt` saved to Drive.

### 0.4 Set up MLflow tracking
- ☐ **Task:** Point MLflow at a Drive-backed store; create experiment `pid-stage4`.
- ☐ **✓ Confirm:** create a dummy run, log one param + one metric, confirm it appears in the MLflow store on Drive. Delete the dummy run.

---

## Phase 1 — Dataset Acquisition & Integrity

### 1.1 Download Gupta PID_Dataset
- ☐ **Task:** Download from `zenodo.org/records/8028570` to Drive.
- ☐ **✓ Confirm:** downloaded file size matches the size reported on Zenodo (±1%). Record the byte count. Checksum (md5/sha256) recorded for reproducibility.

### 1.2 Extract Gupta
- ☐ **Task:** Extract the archive.
- ☐ **✓ Confirm:** **the expected number of annotated sheets is present.** Count *annotated sheets* (not raw image files — the archive also bundles code + trained weights + sample images, so raw image count will exceed 92). Confirm 92 annotated sheets = 72 train + 20 test. Print the count: `assert n_annotated_sheets == 92`. If off, stop and re-extract — do not proceed on a partial dataset.

### 1.3 Download Kaggle P&ID Symbols
- ☐ **Task:** Download from `kaggle.com/datasets/hristohristov21/pid-symbols` (via kagglehub or API token).
- ☐ **✓ Confirm:** file size matches Kaggle's reported ~1.4 GB (±1%). Checksum recorded.

### 1.4 Extract Kaggle
- ☐ **Task:** Extract the archive.
- ☐ **✓ Confirm:** count images and label files. Confirm against the dataset card's stated counts (500 diagrams / 30k tiles / 32 classes / 195k instances). Print counts and `assert` against expected. Confirm all 32 class labels are represented (no missing class).

### 1.5 Verify annotation integrity (both datasets)
- ☐ **Task:** Parse every annotation file; confirm each references an image that exists and vice versa.
- ☐ **✓ Confirm:** zero orphan annotations (annotation with no image), zero unannotated images in the labeled splits. Print `orphans == 0 and unannotated == 0`. Log any mismatch as a hard failure.

### 1.6 Visual spot-check
- ☐ **Task:** Render 5 random Gupta sheets and 5 Kaggle tiles with their bounding boxes overlaid.
- ☐ **✓ Confirm:** by eye, boxes land on actual symbols (not shifted/scaled wrong — catches coordinate-format bugs, e.g. xywh vs xyxy, normalized vs absolute). Save the 10 overlay images to Drive as evidence.

---

## Phase 2 — Data Preparation

### 2.1 Lock the test split
- ☐ **Task:** Separate Gupta's 20 test sheets into a `test/` folder that training code physically cannot read from.
- ☐ **✓ Confirm:** the 20 test sheet IDs are written to a frozen `test_ids.json`. Assert the 72 train sheets and 20 test sheets have **zero overlap**. Assert `len(train)==72 and len(test)==20`. This file is immutable from here on.

### 2.2 Fix the two-part metric (DECISION — resolve before building the harness)
> **This is the crux of the whole benchmark, not a routine checkbox.** Gupta's labels are class-agnostic ("Symbol") — it can score *detection* (is a symbol here?) but **cannot** score *typing* (valve vs instrument). The agent's Stage 4 does typed detection, so typing must be scored somewhere. Decision taken:

- **Part A — Detection recall on Gupta (real data).** Score class-agnostic "did the model find the symbol" against Gupta's real boxes. This is the honest, real-data detection metric. Type correctness is *not* judged here.
- **Part B — Typing accuracy on Kaggle (synthetic data).** Score "given a symbol, is its type correct" against Kaggle's 32 typed classes. This is synthetic, with all synthetic caveats, but it is the only way to test typing without hand-labeling.

- ☐ **Task:** Build two separate scoring paths — detection (Gupta, class-agnostic) and typing (Kaggle, 32-class) — and a documented class map (`classes.json`) reconciling Kaggle's 32 classes to the agent ontology types where they correspond.
- ☐ **✓ Confirm:** written down explicitly that (A) detection is scored on Gupta real, class-agnostic; (B) typing is scored on Kaggle synthetic, 32-class; and that no single dataset scores both. `classes.json` exists and maps every Kaggle class used. **Also record what % of the agent's ontology entity types Kaggle's 32 classes actually cover** — if only ~15 of 32 map to agent types, Part B silently tests typing on a fraction of the real type space, so the coverage number must sit next to the typing score, not just the score alone. The two metrics are reported separately, never averaged into one number that hides which is real and which is synthetic.

### 2.3 Tile the sheets to match agent Stage 3
- ☐ **Task:** Cut sheets into overlapping 1024×1024 tiles (matching the agent's Stage 3 tiling), remapping annotations to tile coordinates.
- ☐ **✓ Confirm:** pick one tile, overlay its remapped boxes, confirm alignment by eye. Assert no annotation was dropped in tiling (sum of per-tile boxes ≥ original box count, accounting for overlap duplicates). Confirm boundary symbols appear in both overlapping tiles.

### 2.4 Convert to each model's expected input format
- ☐ **Task:** Build the eval input format each candidate needs (image + prompt for the VLMs; the expected output schema for scoring).
- ☐ **✓ Confirm:** one sample input round-trips through each model's preprocessor without error. The target/ground-truth format for scoring is fixed and documented.

### 2.5 Per-model output-format prompt engineering (first-class task, not a footnote)
> Getting three different VLMs to emit a **comparable** output format is most of the battle. Qwen wants bbox-JSON, InternVL has its own dialect, Molmo emits points. This is real work and gets its own task.
- ☐ **Task:** For each candidate, engineer the prompt that makes it emit its detections in a parseable, convertible form. Write a per-model parser that maps each model's raw output into the common scoring schema.
- ☐ **✓ Confirm:** each model's parser round-trips ≥ 95% of a 10-sample dev set into the common schema without manual fixup. Parse-failure rate per model is recorded. If a model can't be coaxed above a usable parse rate even with prompt work, flag it — its zero-shot scores will be meaningless (see Phase 4).

### 2.6 Session-budget gate (before committing to a full run)
- ☐ **Task:** Time one model on ~5 tiles; extrapolate to (20 sheets × tiles/sheet × per-tile latency).
- ☐ **✓ Confirm:** a full eval pass over all 20 test sheets fits inside a single Colab session (~12 h) with margin. If it doesn't, batch/checkpoint the eval or reduce candidates per session. Record the estimate before running.

---

## Phase 3 — Metric Harness (build before running any model)

### 3.1 Implement the two scoring functions
- ☐ **Task:** Implement (A) **detection** metrics on Gupta — precision, recall, and **mAP@0.5 / F1** (class-agnostic: is there a symbol here, regardless of type); and (B) **typing** accuracy on Kaggle — per-class typing accuracy over the 32 classes, plus **rare-class recall** (classes with < 20 instances — this belongs here, since Gupta is class-agnostic and has no classes to be rare).
- ☐ **Task (resolve Part B ambiguity):** decide and state how typing is scored — **either** "classify a GT-cropped symbol" (localization handed to the model; easier than reality, isolates pure typing) **or** "detect-then-type on Kaggle" (typing entangled with detection; closer to real use). Pick one; it changes what the typing number means.
- ☐ **✓ Confirm:** feed each scorer a **perfect prediction** → 1.0, **empty** → 0.0, **half-correct** → sane middle. All hold for both scorers or the metric is broken. Confirm the Part B scoring mode (GT-crop vs detect-then-type) is written down.

### 3.2 Define a unified match metric that is fair to BOTH boxes and points
> **Critical:** Qwen/InternVL emit boxes; Molmo emits points. You cannot compute IoU on a point, so a pure box-IoU harness would judge Molmo on a different, easier metric and invalidate the three-way comparison.
- ☐ **Task:** Define one matching rule applied identically to all three candidates. Options: (i) score everyone on **point-in-GT-box hit** (take each prediction's center point; a hit = center falls inside a GT box of the right symbol) — naturally fair to points and boxes; or (ii) derive boxes for Molmo by a fixed rule and score everyone on IoU@0.5. Pick one, apply to all.
- ☐ **✓ Confirm:** the same matching function is called for all three models (no model-specific metric branch). Unit-test it: a point clearly inside a GT box scores a hit; a point clearly outside scores a miss; for the box variant, hand-compute one IoU and assert the function matches. Document which option was chosen and why.

### 3.3 Tile → sheet stitch + NMS dedup (before scoring)
> Input is tiled with overlap, so boundary symbols appear in 2+ tiles. The agent produces **sheet-level** output after cross-tile NMS. Scoring raw per-tile predictions double-counts boundary symbols and distorts precision/recall vs. what the agent actually emits.
- ☐ **Task:** Stitch per-tile predictions back to sheet coordinates and run NMS dedup (mirroring the agent's Stage 4 dedup) before handing predictions to the scorer.
- ☐ **✓ Confirm:** on a sheet with a known boundary symbol, confirm it appears **once** in the stitched output, not twice. Assert stitched count ≤ sum of per-tile counts. Scoring runs on stitched sheet-level predictions, never raw per-tile.

### 3.4 Optional incumbent-comparison column (NOT the pass bar)
> The pass/fail bar is **ground truth** (per your decision). This step adds Claude's Stage-4 output only as an *optional reference column* so you can also see "did the local model match the incumbent," without changing the bar.
- ☐ **Task (optional, do while cloud access is live):** run the cloud agent on the locked 20 test sheets, capture `stage-04` output, score it against the same Gupta/Kaggle GT with the same harness.
- ☐ **✓ Confirm:** if done, Claude's scores appear as one extra row in the comparison table, explicitly labeled "reference, not target." If skipped, note it — the benchmark is still valid because GT is the bar. This never becomes the pass criterion.

### 3.5 Define the IoU/threshold constants
- ☐ **Task:** Fix IoU threshold (0.5) and box format (xyxy absolute) for the box path; fix the point-in-box rule for the point path.
- ☐ **✓ Confirm:** constants written down once, imported everywhere (no magic numbers scattered in code).

### 3.6 Set the tolerance / pass bar
- ☐ **Task:** Define the accuracy threshold that counts as "good enough" for Stage 4 — **as mAP@0.5 or F1 (precision AND recall together), never recall alone** — for detection on Gupta, and typing accuracy on Kaggle, separately. Also set the fallback trigger (the `[X]%` below which a dedicated detector re-enters scope, per spec §1).
- ☐ **✓ Confirm:** the threshold numbers are written down before any model is run. **The detection bar is mAP@0.5 or F1, not recall** — a recall-only bar is gameable: a model emitting thousands of boxes scores near-perfect recall with terrible precision (hallucinated symbols → phantom entities downstream). Confirm precision is part of the pass criterion. Record the numbers.

---

## Phase 4 — Zero-Shot Baseline (all 3 candidates)

> Run each candidate with NO fine-tuning first. **Expectation-setting:** zero-shot VLMs asked to emit structured detections often produce mostly-unparseable output or refuse the format. This phase may only establish "none work zero-shot" — the real comparison is likely post-fine-tune (Phase 5). Do not over-read a zero-shot ranking built on high parse-failure rates.

### 4.1 Load & run Qwen3-VL zero-shot
- ☐ **Task:** Load Qwen3-VL, run the (2.5-engineered) detection prompt over the 20 test sheets' stitched tiles.
- ☐ **✓ Confirm:** model loads within VRAM (no OOM); **record the parse-failure rate** (not required to be 0). Log detection recall + typing accuracy + parse-failure rate to MLflow as `qwen3vl_zeroshot`.

### 4.2 Load & run InternVL3 zero-shot
- ☐ **Task:** Same, InternVL3.
- ☐ **✓ Confirm:** loads, runs, parse-failure rate recorded, metrics logged as `internvl3_zeroshot`.

### 4.3 Load & run Molmo zero-shot
- ☐ **Task:** Same, Molmo (point output → unified metric from 3.2).
- ☐ **✓ Confirm:** loads, runs, parse-failure rate recorded, scored via the **same unified metric** as the others (not a Molmo-only metric), logged as `molmo_zeroshot`.

### 4.4 Compare zero-shot results — with a validity gate
- ☐ **Task:** Pull the 3 runs into one comparison table (detection recall, typing accuracy, parse-failure rate, VRAM/latency).
- ☐ **✓ Confirm:** **first check parse-failure rates.** If all three are high (e.g. > 50% unparseable), record "zero-shot inconclusive — ranking is noise, defer decision to post-fine-tune" and do **not** treat the zero-shot ranking as meaningful. Only if at least one model parses reliably is a zero-shot ranking recorded. Either way, note which model(s) advance to fine-tuning.

---

## Phase 5 — Fine-Tuning (winner, or top 2)

> Layering rule (spec §5): domain-adaptation base is task-neutral; detection is a LoRA on top. Keep them separate.

### 5.1 Build the domain-adaptation training set
- ☐ **Task:** Assemble Kaggle (pretrain volume) + Gupta 72 train sheets into the FT dataset, in the chosen model's training format.
- ☐ **✓ Confirm:** training set contains **only** train data — assert zero test-sheet IDs present (cross-check against `test_ids.json`). Print `intersection(train_ids, test_ids) == empty`. This is the single most important confirmation in the whole checklist — a leak here invalidates every result.

### 5.2 Task-neutral domain fine-tune
- ☐ **Task:** Fine-tune the base on P&ID domain data (QLoRA if VRAM-limited).
- ☐ **✓ Confirm:** training loss decreases over epochs (log curve to MLflow); a checkpoint is saved to Drive every epoch; the run completes without OOM or session timeout. If session times out, confirm resumability from the last checkpoint.

### 5.3 Train the Stage-4 detection LoRA adapter
- ☐ **Task:** On top of the domain base, train the detection-specific LoRA adapter.
- ☐ **✓ Confirm:** adapter file saved separately from the base (confirm you can load base-alone AND base+adapter). Training loss decreases. Checkpoint saved.

### 5.4 Run fine-tuned model on test set
- ☐ **Task:** Run base+detection-adapter over the 20 test sheets.
- ☐ **✓ Confirm:** metrics logged to MLflow as `<model>_finetuned`. Compare against that model's zero-shot run — confirm fine-tuning improved **mAP@0.5 / F1 (not recall alone** — a recall gain with a precision collapse is not an improvement). If the combined metric didn't improve, investigate before proceeding (possible data or format bug).

---

## Phase 6 — Selection & Decision

### 6.1 Final comparison
- ☐ **Task:** Assemble the full comparison: each candidate, zero-shot vs fine-tuned, on the 20 test sheets.
- ☐ **✓ Confirm:** one MLflow table / exported CSV shows all runs with identical metrics and the same test set. Confirm every number came from the locked 20 test sheets (not val, not train).

### 6.2 Apply the pass bar
- ☐ **Task:** Check the winner against the Phase 3.3 threshold.
- ☐ **✓ Confirm:** either (a) winner ≥ threshold → base model selected, record it; or (b) winner < threshold → trigger the spec §1 fallback (dedicated detector re-enters scope) and record that decision. One of these two outcomes is explicitly written down.

### 6.3 Lock the base model
- ☐ **Task:** Record the selected base model, its weights path, config, and scores in the experiment tracker / model registry.
- ☐ **✓ Confirm:** registry row is complete: model name, checkpoint path on Drive, domain-base + detection-adapter paths, test scores, date, `test_ids.json` version. Another person could reload this exact model from the record alone.

### 6.4 Reproducibility check
- ☐ **Task:** Re-run the winner's evaluation from the saved checkpoint in a fresh session.
- ☐ **✓ Confirm:** run the eval with **greedy decoding (temperature 0) and a fixed seed** so the result is deterministic — the reproduced metrics should then be **near-exact** (not "within some vague tolerance"). If they diverge under greedy+fixed-seed, there is real seed/config drift — investigate before trusting the result.

---

## Phase 7 — Handoff to Downstream

### 7.1 Document what the base is for reuse
- ☐ **Task:** Write down that this base (without the detection adapter) is the shared base for the reasoning stages (10.5, 13, 12, etc.), per spec §5.
- ☐ **✓ Confirm:** the handoff note states clearly: which checkpoint is the task-neutral base, that the detection LoRA is Stage-4-only, and where both live. Downstream stage work can pick this up without re-asking.

### 7.2 Log open risks + the inherent metric limitation
- ☐ **Task:** Record any Stage-4-specific caveats surfaced (e.g. Molmo won on detection but is unproven on reasoning; rare-class recall weak; tiling artifacts).
- ☐ **Task (mandatory limitation note):** Record the one thing the two-part metric structurally cannot prove: **find-and-type-correctly jointly on real drawings is never tested**, because no real typed ground truth exists. Part A proves detection on real sheets; Part B proves typing on *synthetic* symbols. A reader seeing "detection 0.90 / typing 0.85" must NOT conclude the model does typed detection at ~0.85 on real sheets — **typing on real data is untested, and the synthetic→real gap (the 73%→27% collapse) bites hardest exactly there.**
- ☐ **✓ Confirm:** both the caveats and the explicit limitation sentence are written into the tracker, so nobody over-reads the two numbers and so the reasoning-stage benchmarks account for them.

---

## Master Gate (before declaring Stage 4 done)

- ☐ All confirmations above passed.
- ☐ Test split provably never seen during training (5.1 confirmation).
- ☐ Metric harness sanity-checked on perfect/empty/half inputs (3.1).
- ☐ Base model selected against a pre-registered threshold (6.2).
- ☐ Result reproducible from saved checkpoint (6.4).
- ☐ Base + adapter locked and documented for reuse (6.3, 7.1).

**Only when every box is ticked is the base model decision trustworthy and the project ready to move to Stage 1.5 / downstream stages.**
