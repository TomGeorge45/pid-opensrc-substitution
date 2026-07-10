# Stage 4 Symbol Detection — Candidate Modules

Model loading/prompting/parsing code for the Stage 4 VLM bake-off (Qwen3-VL, InternVL3,
Molmo2-O-7B), extracted from `notebooks/stage4/Stage4_Phase2_ModelSetup.ipynb` into plain
`.py` modules so fixes don't require re-opening/re-running the notebook from GitHub every
time — just `git pull` + re-import.

## Colab quickstart

Run once per fresh runtime:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/TomGeorge45/pid-opensrc-substitution.git /content/pid-ml
# on subsequent runs in the same session, use this instead:
# !cd /content/pid-ml && git pull

import sys
sys.path.insert(0, '/content/pid-ml/src')

!pip install -q torch transformers accelerate pycocotools supervision kagglehub kaggle \
    qwen-vl-utils einops timm
```

Then, per candidate:

```python
from stage4_symbol_detection import qwen_candidate, data_utils, eval_harness

processor, model = qwen_candidate.load()
img, img_path, label_path, gt_count = data_utils.load_sample()

raw_text, latency = qwen_candidate.run(processor, model, img)
detections, error = qwen_candidate.parse(raw_text)

dev_set = eval_harness.build_dev_set()
results, parse_rate = eval_harness.run_parse_check(
    dev_set,
    run_fn=lambda img: qwen_candidate.run(processor, model, img),
    parse_fn=qwen_candidate.parse,
)
```

InternVL3 is the same shape (`internvl_candidate.load()` / `.run()` / `.parse`). Molmo2
needs `!pip install -q transformers==4.57.1` **and a runtime restart** before `load()` — see
`molmo_candidate.py`'s module docstring — and its `parse()` takes `(text, img_w, img_h)`
since its coordinate format is normalized, unlike the other two.

## Updating after a fix

No need to re-open the notebook from GitHub. Just:

```python
!cd /content/pid-ml && git pull
```

then re-run your `import`/`load()` cells (Python caches modules per-process, so if you only
`git pull` without restarting, use `importlib.reload(qwen_candidate)` instead of a fresh
`import` to pick up code changes without a full runtime restart).
