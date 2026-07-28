"""REAL Molmo2-O-7B per-class pointing wrapper for the `pnid-extraction-agent` local stack
(Extraction_Agent_Local_Plan.md §3.A/§3.B Molmo2 slot; Phase 2, GPU).

Load recipe copied VERBATIM from this project's PROVEN Molmo2 notebooks (this project
switched from the original `Molmo-7B-D-0924` checkpoint to Molmo2 specifically because the
2024-era checkpoint's custom `trust_remote_code` class hit 5 escalating incompatibilities
with current `transformers` — Molmo2 uses the standard `AutoModelForImageTextToText`/
`generate()` pattern, per CLAUDE.md's candidate table):
  - `notebooks/e2e_harness/ArmL_Molmo2_Qwen_Mixed_GPUOnly.ipynb` §5 ("Load Molmo2-O-7B"):
    `MOLMO_MODEL_ID = "allenai/Molmo2-O-7B"`, `AutoProcessor.from_pretrained(MOLMO_MODEL_ID,
    trust_remote_code=True, dtype="auto")`, `AutoModelForImageTextToText.from_pretrained(
    MOLMO_MODEL_ID, trust_remote_code=True, dtype="auto", device_map="cuda")`.
  - Cross-checked against `notebooks/all_vlm_stages_benchmarking/
    Stage105_SkidMatrix_Molmo2_Qwen_Adapters_GPUOnly.ipynb` §5 (identical recipe) and
    `src/stage4_symbol_detection/molmo_candidate.py`'s `load()` (identical again).
  - `apply_chat_template` message order for Molmo2 is TEXT-then-IMAGE (opposite of Qwen's
    IMAGE-then-TEXT order used in `qwen_generate.py`) — confirmed in all 3 notebooks'
    `molmo_generate(image, prompt_text)` closures:
    `messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text},
    {"type": "image", "image": image}]}]`.

Tiling/upscale config: **512px tiles / 102px overlap / 2x upscale / autocontrast-grayscale
"enhance"** is the ONLY validated Molmo2 config in this project (CLAUDE.md: "Molmo2-O-7B ...
Unproven on reasoning stages"; the config itself — tile=512, overlap=102, upscale=2,
enhance=True — is `notebooks/all_vlm_stages_benchmarking/Stage4_Detection_GPT55_vs_Molmo2.ipynb`'s
recorded F1=0.628 run, vs. F1=0.434 for the unenhanced/no-upscale 1024px production-grid
config). Do NOT silently change tile size or upscale factor — that would be re-deriving an
unvalidated config, exactly what CLAUDE.md/this plan warns against.

`<points>` parsing reuses `e2e_bench/backends/parse_molmo.py`'s `parse_molmo_points` — the
same parser behind the recorded F1=0.628 run — not reimplemented here.

COORDINATE SPACE (read before using): `img` passed to `molmo_point_classes` MUST be the
array `molmo_render.molmo_render_page(...)` returns (or a numpy array derived from it that
you can map back to it at a KNOWN scale) — i.e. the SAME coordinate space `extract_page`
itself renders internally for whichever pipeline branch you're feeding (see
`molmo_render.py`'s docstring for why `triage_page`/`work_zoom`/`render_page`, in that exact
order, is required — Phase 1c fix 1, rotated-page correctness). This module's tiling and
tile-origin math never rescales `img` itself (only per-tile crops are upscaled for the model,
then the upscale is divided back out before adding the tile origin) — so every returned
point is in EXACTLY `img`'s own pixel space, full stop. Passing in a differently-scaled
render (e.g. the 150dpi `render_pdf_page` used elsewhere in this project purely for the
PaddleOCR sanity check) would silently produce points in the wrong coordinate space; this is
the exact class of bug Phase 1c fix 1 was written to eliminate.

NOT tested end-to-end here — no GPU in this environment.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from e2e_bench.backends.parse_molmo import parse_molmo_points  # noqa: E402
from extraction_local.molmo_candidates import _dedup_points  # noqa: E402

MOLMO_MODEL_ID = "allenai/Molmo2-O-7B"

# Validated config (Stage4_Detection_GPT55_vs_Molmo2.ipynb, F1=0.628) — do not change
# silently, see module docstring.
DEFAULT_TILE = 512
DEFAULT_OVERLAP = 102
DEFAULT_UPSCALE = 2.0
DEFAULT_ENHANCE = True
DEFAULT_MAX_NEW_TOKENS = 600  # ArmL_Molmo2_Qwen_Mixed_GPUOnly.ipynb's MOLMO_MAX_NEW_TOKENS

# Default class list for the extraction-agent Molmo2 slot (§3.A/§3.B). This is a NEW list
# for THIS repo's extraction-agent work — it is NOT the same as the intelligence-agent's
# `e2e_bench.ontology.entity_types()` 6-way split used by
# `ArmL_Molmo2_Qwen_Mixed_GPUOnly.ipynb`/`Stage105_...ipynb` (that ontology belongs to a
# different agent/benchmark; reusing it here would silently import assumptions from the
# wrong pipeline). Phrased in the same short-imperative "point to every X" style those
# notebooks validated works best for Molmo2 (`Stage4_Phase4_MolmoZeroShot.ipynb`'s v3
# prompt rationale: short, positive, no negated clauses).
DEFAULT_CLASSES: List[str] = [
    "valve", "instrument bubble", "pump", "vessel or tank",
    "off-page connector", "equipment",
]

_PROMPT_TEMPLATE = "Point to every {noun} in this P&ID tile."


def _prompt_for_class(cls: str) -> str:
    return _PROMPT_TEMPLATE.format(noun=cls)


def load_molmo_model(model_id: str = MOLMO_MODEL_ID) -> Tuple[Any, Any]:
    """Real `from_pretrained` load, verbatim recipe from
    `ArmL_Molmo2_Qwen_Mixed_GPUOnly.ipynb` §5 / `Stage105_SkidMatrix_...ipynb` §5 /
    `src/stage4_symbol_detection/molmo_candidate.py::load()`: `trust_remote_code=True`,
    `dtype="auto"`, `device_map="cuda"`. Requires a CUDA GPU — no silent CPU fallback.

    Returns (model, processor).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "load_molmo_model() requires a CUDA GPU (dtype='auto' + device_map='cuda', "
            "same recipe as ArmL_Molmo2_Qwen_Mixed_GPUOnly.ipynb) — none detected. This is "
            "Phase 2 GPU code; run it in the Colab notebook, not on the Mac."
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, dtype="auto")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", device_map="cuda"
    )
    return model, processor


def _compute_tile_grid(img_w: int, img_h: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Same stride-walking grid as `Stage4_Detection_GPT55_vs_Molmo2.ipynb`'s
    `compute_tile_grid` (stride = tile - overlap, last row/col clipped to image bounds)."""
    stride = max(1, tile - overlap)
    tiles: List[Tuple[int, int, int, int]] = []
    y0 = 0
    while y0 < img_h:
        y1 = min(y0 + tile, img_h)
        x0 = 0
        while x0 < img_w:
            x1 = min(x0 + tile, img_w)
            tiles.append((x0, y0, x1, y1))
            x0 += stride
        y0 += stride
    return tiles


def _enhance_tile(pil_img):
    """Verbatim `enhance_tile` from `Stage4_Detection_GPT55_vs_Molmo2.ipynb`: autocontrast
    on the grayscale conversion, then back to RGB — part of the validated 512/102/2x/
    enhance=True config, not a cosmetic add-on."""
    from PIL import ImageOps

    return ImageOps.autocontrast(pil_img.convert("L")).convert("RGB")


def _molmo_generate_raw(model, processor, pil_image, prompt_text: str, max_new_tokens: int) -> str:
    """One Molmo2 generate() call. TEXT-then-IMAGE message order (see module docstring) —
    verbatim from the proven notebooks' `molmo_generate` closures."""
    import torch

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text}, {"type": "image", "image": pil_image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def molmo_point_classes(
    model: Any,
    processor: Any,
    img: np.ndarray,
    classes: List[str] = None,
    *,
    tile: int = DEFAULT_TILE,
    upscale: float = DEFAULT_UPSCALE,
    overlap: int = DEFAULT_OVERLAP,
    enhance: bool = DEFAULT_ENHANCE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    dedup: bool = True,
) -> Dict[str, List[Tuple[float, float]]]:
    """Run Molmo2 per-class "point to every X" pointing over `img`, tiled at the validated
    512px/102px-overlap/2x-upscale/enhance=True config (do not change these defaults
    silently — see module docstring), one separate generate() call per (tile, class) pair
    (matches this project's established "N separate pointing calls per tile, Molmo2 has no
    native per-point class label" pattern — `ArmL_Molmo2_Qwen_Mixed_GPUOnly.ipynb` decision
    1).

    `img`: RGB numpy array, MUST be `molmo_render.molmo_render_page(...)`'s output (or
    something at that EXACT known scale) — see module docstring's COORDINATE SPACE section.

    Returns `{class_label: [(x, y), ...]}` with (x, y) in `img`'s own full-page pixel space
    (tile-local upscaled model coords -> divided back by `upscale` -> tile origin added).
    `dedup=True` (default) collapses near-duplicate points across overlapping tiles using
    the same `_dedup_points(min_dist=radius/2.0)` convention `molmo_candidates.py`/
    `run_extraction_local.py`'s Molmo wrapper already use elsewhere in this project (radius
    hardcoded to the Molmo synthetic-token pairing default, 120px, matching
    `Extraction_Agent_Local_Plan.md` Phase 1c fix 2) — pass `dedup=False` if the caller wants
    to do its own deduping instead (e.g. against a different downstream radius).
    """
    from PIL import Image

    if classes is None:
        classes = DEFAULT_CLASSES

    H, W = img.shape[:2]
    tiles = _compute_tile_grid(W, H, tile=tile, overlap=overlap)

    points_by_class: Dict[str, List[Tuple[float, float]]] = {cls: [] for cls in classes}

    for (x0, y0, x1, y1) in tiles:
        crop_arr = img[y0:y1, x0:x1]
        if crop_arr.size == 0:
            continue
        pil_tile = Image.fromarray(crop_arr)

        if upscale and upscale != 1.0:
            pil_tile = pil_tile.resize(
                (int(pil_tile.width * upscale), int(pil_tile.height * upscale)),
                Image.LANCZOS,
            )
        if enhance:
            pil_tile = _enhance_tile(pil_tile)

        tile_w_up, tile_h_up = pil_tile.size

        for cls in classes:
            prompt = _prompt_for_class(cls)
            raw_text = _molmo_generate_raw(model, processor, pil_tile, prompt, max_new_tokens)
            outcome = parse_molmo_points(raw_text, tile_w_up, tile_h_up, entity_type=cls)
            if outcome.parse_failed or outcome.value is None:
                continue  # honest failure mode: skip this (tile, class) pair's points

            for det in outcome.value:
                bx0, by0, bx1, by1 = det.bbox_tile
                cx_up, cy_up = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
                # Undo upscale, then undo tile offset (Phase 1c coordinate-space fix
                # applies to the RENDER step; this is the analogous per-tile undo for
                # Molmo's own tiling, same principle: never leave a scale factor un-undone).
                x = cx_up / upscale + x0
                y = cy_up / upscale + y0
                points_by_class[cls].append((x, y))

    if dedup:
        deduped = _dedup_points(points_by_class, min_dist=120.0 / 2.0)
        result: Dict[str, List[Tuple[float, float]]] = {cls: [] for cls in classes}
        for (cls, x, y) in deduped:
            result[cls].append((x, y))
        return result

    return points_by_class
