"""REAL Qwen3-VL-8B `generate_fn` — the GPU counterpart to Phase 1's fake
`generate_fn`s in `run_extraction_local.py` (`_fake_generate_fn`/`_fake_generate_fn_ocr`).

Load recipe and `apply_chat_template`/`generate()` call shape copied VERBATIM from this
project's own PROVEN, already-working Qwen3-VL notebooks — not reinvented:
  - `notebooks/e2e_harness/ArmL_QwenVL_FullStack_GPUOnly.ipynb`, cell 13 ("## 5. Load
    Qwen3-VL-8B (base, no adapter)") — `AutoProcessor`/`AutoModelForImageTextToText`,
    `dtype=torch.bfloat16`, `device_map="cuda"`, and the `qwen_generate(image, prompt_text,
    max_new_tokens=4096)` closure (chat-template message shape, greedy `generate()`, decode
    only the newly-generated tail).
  - Cross-checked against `notebooks/e2e_harness/ArmL_Molmo2_Qwen_Mixed_GPUOnly.ipynb`
    (§11/§12, `make_qwen_generate_fn`) and
    `notebooks/all_vlm_stages_benchmarking/Stage105_SkidMatrix_Molmo2_Qwen_Adapters_GPUOnly.ipynb`
    (§5/§9) — all three use the identical recipe: `QWEN_MODEL_ID =
    "Qwen/Qwen3-VL-8B-Instruct"`, `messages = [{"role": "user", "content": [{"type": "image",
    "image": <PIL.Image>}, {"type": "text", "text": prompt_text}]}]`,
    `processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt").to(model.device)`, then
    `model.generate(**inputs, max_new_tokens=..., do_sample=False)`, decoding only
    `out[0][inputs["input_ids"].shape[1]:]`.

`transformers==4.57.1` is the pinned version all three notebooks use (known-good for
`Qwen3-VL-8B-Instruct` + `AutoModelForImageTextToText`/`peft`) — install it in the Colab
notebook before calling `load_qwen_model()`.

NOT tested end-to-end here — no GPU in this environment. This module has no torch import
required to be exercised until `load_qwen_model()`/the returned `generate_fn` is actually
called, matching this project's established "generate_fn is injected, no torch import at
module scope for the parts that don't need it" convention (see `qwen_call_llm.py`), except
the loader function itself obviously does need torch/transformers at call time.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Callable, List, Optional, Tuple

QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

GenerateFn = Callable[..., str]  # generate_fn(prompt, images, max_new_tokens=None) -> str


def load_qwen_model(model_id: str = QWEN_MODEL_ID) -> Tuple[Any, Any]:
    """Real `from_pretrained` load, verbatim recipe from `ArmL_QwenVL_FullStack_GPUOnly.ipynb`
    cell 13: `AutoProcessor.from_pretrained(model_id)` +
    `AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16,
    device_map="cuda").eval()`. Requires a CUDA GPU (bf16 on `device_map="cuda"`) — will
    raise if none is present; this is intentional, no silent CPU fallback (an 8B VLM on CPU
    is not a usable substitute for what this harness measures).

    Returns (model, processor), matching the notebook cell's call order so the notebook can
    do `model, processor = load_qwen_model()` directly.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "load_qwen_model() requires a CUDA GPU (bf16 + device_map='cuda', same recipe "
            "as ArmL_QwenVL_FullStack_GPUOnly.ipynb) — none detected. This is Phase 2 GPU "
            "code; run it in the Colab notebook, not on the Mac."
        )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    return model, processor


def _b64_png_to_pil(b64_str: str):
    """Decode one base64-encoded PNG string into a PIL Image. This is the established
    image convention used elsewhere in this project's Qwen call sites (`qwen_call_llm.py`'s
    docstring/contract: `images: list[str_b64]`) — decode here, once, before building the
    chat-template message, same as the ArmL notebooks do when they hand a PIL.Image (not a
    base64 string) to `apply_chat_template`'s `{"type": "image", "image": ...}` content
    entry."""
    from PIL import Image

    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_qwen_generate_fn(
    model: Any,
    processor: Any,
    max_new_tokens_default: int = 4096,
) -> GenerateFn:
    """Build a `generate_fn` matching the EXACT signature `qwen_call_llm.build_qwen_call_llm`
    calls it with (see that file's `call_llm` closure):

        generate_fn(full_prompt, imgs)                          # no cap configured
        generate_fn(full_prompt, imgs, max_new_tokens=cap)      # cap configured

    i.e. `max_new_tokens` is passed as a keyword arg ONLY when `build_qwen_call_llm` was
    given a `max_new_tokens_cap` — this function must accept BOTH call shapes, matching the
    existing fakes' `def _fake_generate_fn(prompt, images, max_new_tokens=None)` signature
    (`run_extraction_local.py`) so it is a drop-in replacement for either fake.

    `images` is a list of base64-encoded PNG strings (this project's established Qwen image
    convention) — decoded into PIL Images here before constructing the chat-template
    message, exactly like the ArmL notebooks' `qwen_generate(image, prompt_text, ...)`
    closures (which take an already-decoded PIL.Image directly since their harness never
    base64-encodes in the first place; the encode/decode round-trip only exists here
    because `qwen_call_llm.py`'s contract is base64 strings, to match the async/JSON-shim
    convention already established for this project's other local-model call_llm shims).

    Greedy decoding (`do_sample=False`), decode only the newly-generated tail
    (`out[0][inputs["input_ids"].shape[1]:]`) — verbatim from the proven notebook recipe.
    """
    import torch

    def generate_fn(prompt: str, images: List[str], max_new_tokens: Optional[int] = None) -> str:
        n_tokens = max_new_tokens if max_new_tokens is not None else max_new_tokens_default

        pil_images = [_b64_png_to_pil(b64) for b64 in (images or [])]

        content: List[dict] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            # repetition_penalty ONLY -- greedy decoding (do_sample=False) with no repetition
            # guard at all is a classic setup for degenerate repetition loops on long
            # generations -- confirmed live, 2026-07-23 (PartB_Qwen_RelationRun_GPUOnly.ipynb
            # smoke gate): 183/183 "tags" turned out to be the SAME entity ("PSV-0300C")
            # repeated verbatim with incrementing ids, not real diverse content.
            #
            # `no_repeat_ngram_size` was tried alongside this and REMOVED -- confirmed live
            # in the SAME notebook session: it's a hard block on any repeated N-token
            # sequence anywhere in the output, which is fundamentally wrong for JSON-array
            # output, where every tag entry legitimately repeats identical structural
            # phrases (`"type": "equipment",\n      "word_ids": [`, etc.) -- adding it
            # collapsed a real, diverse 183-tag response down to just 1 tag total (the model
            # got blocked from completing a second entry's boilerplate and gave up). Do not
            # re-add no_repeat_ngram_size for this task; it's the wrong tool for
            # structured/formatted output specifically, not merely a tuning question.
            # `repetition_penalty` alone is a soft, per-token probability nudge, not a hard
            # block, so it doesn't have this failure mode.
            out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False,
                                 repetition_penalty=1.15)

        gen = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(gen, skip_special_tokens=True).strip()

    return generate_fn
