# Stage 4 — Symbol Detection Benchmarking · Task Checklist


**Revision:** 3.0 — precision in pass bar (mAP/F1 not recall), real-typing limitation logged, rare-class + coverage + Part-B-mode slips fixed
**Goal:** Pick the base VLM (Qwen3-VL vs InternVL3 vs Molmo) by benchmarking symbol detection against Gupta ground truth, then domain-fine-tune the winner and train a detection LoRA. This decision unlocks every downstream stage.

**How to use:** every task has a paired **✓ Confirmation** — an explicit check that must pass before moving on. Do not proceed to the next task until its confirmation passes. Confirmations are concrete (a count, a file, a number), never "looks fine."

**Legend:** ☐ not started · ◐ in progress · ☑ done + confirmed

---

## 0. Infrastructure, Setup, and Model Benchmarking History (read this first)

This section is a standing reference — updated as the project evolves, not a one-time checklist item. It covers *how* this project actually runs (tooling, why it's set up this way) and *what has been tried so far* (every model tested, how it was fine-tuned, and how results changed across iterations). Read this before touching Phase 0 below, since some of Phase 0's original assumptions (Drive-based persistence) have since been superseded.

### 0.A The actual working setup: VS Code + Colab + Hugging Face Hub

**Three pieces, each doing a different job:**

- **VS Code, with the Microsoft "Jupyter" extension (`ms-toolsai.jupyter`), local on the Mac** — where notebooks (`.ipynb` files) are authored, reviewed, and version-controlled. This extension is what renders a notebook as editable cells inside VS Code at all (without it, a `.ipynb` is just a JSON file). Nothing executes here — there's no usable GPU on the local machine, and this local copy is **never connected to a live kernel**. This is the source of truth for code, not for execution.
- **Google Colab (separate browser tab, GPU runtime)** — where everything actually *runs*. Training and inference happen on a Colab Pro+ GPU runtime (A100 preferred over H100 — the two are similar value once training doesn't need to finish in one sitting, since a per-hour price gap that roughly tracks a per-hour speed gap is a wash; A100 wins as the safer/cheaper default whenever there's no single-session urgency). Code is copied from VS Code into Colab to run, then results/checkpoints are pulled back out.
- **Hugging Face Hub (private repos, personal account)** — the shared storage layer for both raw datasets and trained checkpoints. Two repos: a dataset repo (`timthy45/pnid-extraction-datasets` — Gupta/Kaggle/PID2Graph zips, benchmark result CSVs) and a model repo (`timthy45/qwen3vl-pnid-domain-base` — every adapter checkpoint, pushed periodically during training).

**How these three actually connect — and the one thing that trips people up every time:** there is **no live link** between the VS Code copy of a notebook and whatever is running in the Colab browser tab. They are two independent copies of the same file. The real flow is:

```
VS Code (.ipynb on disk)  --[manual copy/paste or re-upload]-->  Colab (live GPU kernel)
                                                                        |
                                                                        v
                                                          huggingface_hub calls (token from Config cell)
                                                                        |
                                                                        v
                                                        Hugging Face Hub (dataset repo + model repo)
```

Editing a cell in VS Code **does nothing to an already-running Colab session** until that cell is manually re-copied over — the Colab tab keeps executing whatever it loaded last, oblivious to local edits. This has caused real, repeated confusion: at one point, having the notebook open in VS Code while it was being edited via automation caused VS Code's own in-memory buffer to *resave stale content back over the edit* — including reintroducing a real API token that had just been scrubbed out. **Lesson for anyone continuing this work: after editing a notebook file, always re-copy the changed cell(s) into the actual live Colab tab yourself — don't assume it updates automatically, and don't assume the file on disk is what's currently running in Colab.**

**Why not Google Drive (the obvious alternative)?** Tried first, ruled out for a structural reason, not a preference: `drive.mount()` uses an OAuth flow that's hard-bound to the account that owns the Colab runtime. Since the GPU quota (Colab Pro+) lives on a different account than the one doing the work, every mount attempt sent a 2FA verification code to the *runtime owner's* phone — regardless of browser, device, or incognito mode. This is not fixable from inside the notebook, and it specifically breaks any unattended/overnight run (nobody there to approve the prompt). Hugging Face Hub's alternative — a single bearer token, no OAuth, no 2FA, no account coupling — decouples "who owns the compute" from "who owns the storage" entirely. This is the reason, not a stylistic choice; don't reintroduce Drive as a shortcut later without solving the 2FA problem first.

**Practical consequences of this setup, learned the hard way:**
- **HF's storage backend ("Xet") occasionally serves a broken signed URL** (`403 Forbidden: SignatureError: invalid key pair id`), confirmed server-side (persists even with the `hf_xet` client package fully uninstalled). Every download in every notebook now goes through a hardened retry wrapper: up to ~20 attempts, exponential backoff (10s→90s), and critically *no* `force_download=True` on retries (that wastes bandwidth restarting a good partial download — HF mints a fresh signed URL on every call regardless, so a plain retry gets the same benefit for free). Every real occurrence of this bug has cleared within a handful of retries.
- **ModelScope (Alibaba's model hub) was tried as a bypass for the above and rejected.** It hosts the same public Qwen weights on different infrastructure (so the Xet bug can't happen there), but on this network it measured ~1Mbps vs ~200Mbps on HF — a 200x penalty that makes it strictly worse despite dodging the bug. Don't re-add it.
- **Every checkpoint push to a fixed HF path is a new git commit under the hood.** Historical LFS blobs stay counted against storage quota even though the file listing only shows the current version — this silently ate through a 100GB free-tier quota (211 commits, 82.5GB on one repo) before being caught. Fixed by deleting and recreating the repo with just the checkpoints actually needed; worth periodically checking commit count (`HfApi().list_repo_commits(...)`) if storage warnings reappear.
- **Compute-cost discipline (standing directive from Tom's manager, not yet fully implemented in every notebook):** do CPU-only work (dataset download/extraction, OCR, building training/eval example pools) locally on the Mac, not inside the paid GPU Colab session — then zip and upload the prepared result to HF so Colab only does the GPU-dependent steps (model load, train/infer, push results) before immediately calling `from google.colab import runtime; runtime.unassign()` to release the GPU. Apply this to any new notebook going forward; retrofit older ones opportunistically.

### 0.B How to set this up from scratch

1. **Clone/open this repo in VS Code.** All notebooks live under `notebooks/`; this is where they're written and edited.
2. **Open a separate browser tab at colab.research.google.com**, connect to a GPU runtime (A100 preferred; check VRAM with `!nvidia-smi` — 80GB comfortably fits an 8B model plus multiple LoRA adapters loaded simultaneously).
3. **Create two private HF repos** under the working personal account (not the org/boss's account, to keep storage independent of the compute-quota owner): one `dataset` repo for raw data + result CSVs, one `model` repo for checkpoints.
4. **Generate an HF token with write access.** Paste it only into the notebook's Config cell when running in Colab — **never commit a real token to this repo.** Every notebook uses a placeholder (`"paste-your-hf-token-here"`) by default; always re-check for a leaked real token (`grep -oE "hf_[A-Za-z0-9]{20,}"`) before any commit, since editors left open on a notebook can silently resave a pasted token back into the tracked file.
5. **Use `huggingface_hub` for all data movement** (`hf_hub_download`, `snapshot_download`, `HfApi().upload_file`/`upload_folder`) instead of `drive.mount()` — wrapped in the hardened retry pattern described above.
6. **Never reach for ModelScope or Drive as a fix** for a slow/failed download — both have been tried and are proven net negatives on this specific setup.
7. **Follow the compute-cost discipline** above for any new notebook: CPU-prep locally → zip → HF → GPU-only Colab session → `runtime.unassign()`.

### 0.C Models tested, how they were fine-tuned, and how results changed

Two separate tracks exist in this project: **Stage 4 (symbol detection)**, which is genuinely in-scope per this document's charter, and **the reasoning-stage LoRA work** (stages 1/2/5/10.5/12/13), which uses Qwen3-VL-8B as a shared base and is being developed alongside Stage 4 at Tom's explicit direction even though it sits outside this document's original "Stage 4 only" framing. Both are recorded here since they share the same infrastructure and models.

#### Stage 4 (symbol detection) candidates

| Model | Role | Fine-tuning | Result |
|---|---|---|---|
| **Molmo2-O-7B** | Stage 4 detection candidate (native pixel-pointing) | Zero-shot only so far, at multiple tiling/enhancement configs | Baseline config: full-20-sheet zero-shot F1 = **0.434** vs the incumbent cloud agent's **0.380** (both still below the 0.70 pass bar). An improved config (512px tiles, 2× upscale, autocontrast enhancement) subsequently **beat the incumbent** on the full 20 test sheets — the exact number for that run should be pulled fresh from `stage4-status.html` rather than quoted from memory here, since it wasn't directly re-verified while writing this section. |
| **GPT-5.5 (low reasoning)** | Cloud/API reference point for detection | Not fine-tuned (prompted zero-shot) | At one checkpoint in testing, GPT-5.5 was recorded as the **new best zero-shot detector** on the full 20 sheets, ahead of Molmo2's baseline config at the time. Whether it still leads after Molmo2's improved config needs a direct side-by-side re-check. |
| **NVIDIA Nemotron-Nano-VL-8B** | Considered, then **ruled out** | Zero-shot probe only | Given a symbol-grounding task, it emits DocVQA-style text-line/layout bands (its training distribution) instead of boxes on symbols — confirmed visually (~90 wall-to-wall horizontal strips, zero boxes on actual symbols). Kept only as a plausible candidate for pure text-reading stages, not detection. |
| **InternVL3 (8B)** | Original spec candidate | Not deeply tested in this repo | Named in the original 3-way bake-off design (`PID_Local_Substitution_Spec.md` §5) but no benchmark run for it is recorded in this session's history — don't assume it's been ruled in or out. |
| **Claude (claude-sonnet-4-6)** | Incumbent production reference | N/A (already deployed) | Used as the "did we beat the current cloud agent" reference column — F1 0.380 on the full 20 sheets, the number Molmo2/GPT-5.5 are measured against. |

#### Reasoning-stage LoRA adapters (Qwen3-VL-8B-Instruct, shared base for stages 1/2/5/10.5/12/13)

**v1 — mixed-task domain adaptation, all-linear LoRA.** One adapter trained on four tasks at once: OCR/tag-listing, symbol counting, typed symbol summary (Kaggle), and connectivity Q&A (PID2Graph). `target_modules="all-linear"` (touches the vision tower too). Counting collapsed to "always answer 0" (100% zero-answers against a 12% true zero-rate) and full-tile tag listing dropped from the base's 39% to 2%.

**v2 — same mixed-task approach, targeted fixes.** Capped zero-answer tiles at 15% of training data and varied phrasing to fix v1's counting collapse (0% zero-answers afterward, MAE improved 15.0→12.3). Connectivity and typing held steady. But **tag reading was still destroyed** — even with a redesigned task (clean single-tag crops instead of full-tile listing), the adapter scored 0% on the exact crops it trained on, while the untouched base model read 72% of the same crops. This was the finding that mattered: two different training rounds, two different reading-task designs, same result — strongly implicating the training process itself (LoRA touching the vision tower with gradients that never rewarded glyph precision), not the task design.

| Task (metric) | Base | v1 | v2 |
|---|---|---|---|
| Relation accuracy (real PID2Graph GT) | 56–77%* | 90% | 88% |
| Typed-summary class F1 | 0.00 | 0.36 | 0.34 |
| Count: answered "0" (truth 12%) | 6% | 100% ✗ | 0% ✓ |
| Count MAE | 16.3 | 15.0 | 12.3 |
| Tag reading (single-tag crops) | 72% | — | 0% ✗ |
| Full-tile tag listing (v1's task) | 39% | 2% ✗ | 2% ✗ |

*Base relation accuracy varies with how many answers were decidable (34–37 of 50 undecided without a clean yes/no prompt).

**Conclusion drawn from v1/v2: general, mixed-task training is destructive** — a fix for one task reliably damaged an unrelated one. This motivated the shift to **v3: dedicated, per-stage adapters, language-only LoRA** (vision tower explicitly excluded from `target_modules`, discovered at runtime by inspecting the loaded model and filtering out anything with "visual"/"vision"/"image_tower" in its name — the direct fix for the all-linear hypothesis above).

**v3 results (all measured at n=120, or n=65 for the naturally-smaller OCR pool — not the noisy n=25 samples used earlier in the project, which were shown to swing a "win" into a "loss" on a single flipped answer):**

| Task | GPT-5.5-low | Qwen base (no adapter) | Dedicated v3 adapter | Old v2 general adapter (n=25, unverified at scale) |
|---|---|---|---|---|
| Entity validation (stage 13) — "is a real symbol here?" | 66.7% | 45.0% (below chance) | **89.2%** — fully trained, 3/3 epochs | 35.0% (below chance) |
| Relation validation (stage 12/10.5) — "are these connected?" | 72.5% | 80.0% | **89.2%** — partially trained, paused mid-epoch-0 at step ~44,853/64,911 | 84.0% (noisy — a single-question margin) |
| Text extraction (stage 5) — OCR A/B tie-break | 98.5% | **100.0%** | n/a — no adapter built; base already wins | 66.7% (mostly unparseable answers, low confidence) |

Two adapter-specific notes: (1) **v3-stage13's win came from deliberate absence supervision** — three decoy types (empty region, shifted near-miss, wrong-size box) mixed 50/50 with real boxes during training, specifically to prevent the "always say yes" bias that made the old v2 adapter score *below chance* on this exact task. (2) **v3-relation's eval, while real, isn't fully rigorous yet** — it's seed-disjoint from training (different random seed, same source data trees) rather than file-disjoint like the Gupta-based tasks; a genuinely held-out file split is still owed if this number needs to support a bigger claim later.

**Where things stand:** v3-stage13 is finished and is the clear best model on its task, beating both GPT-5.5 and the old general adapter. v3-relation already clears its adoption bar (beats prompted-base by a real margin at a trustworthy sample size) despite being only a third of the way through training — resuming its remaining two epochs is a known, not-yet-executed next step. Text extraction needs no adapter at all — the untouched base model is already the best performer, and past attempts to fine-tune it only made it worse.

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
