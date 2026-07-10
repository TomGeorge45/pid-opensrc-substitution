"""Molmo2-O-7B candidate: load, prompt, parse.

REQUIRES `transformers==4.57.1` — run this before importing this module in a fresh runtime:

    !pip install -q transformers==4.57.1

Then restart the runtime before loading. See Stage4_Checklist_Status.md item 2.5 for why:
the original Molmo-7B-D-0924 candidate was dropped after 5 escalating incompatibilities with
newer transformers versions (its custom trust_remote_code class predates transformers v5's
generation-internals restructuring). Molmo2-O-7B uses the standard AutoModelForImageTextToText
pattern like the other two candidates, but its own model card pins this specific version.

Deferred experiment (not yet done): Qwen3-VL needed transformers==4.57.0 at its own release;
since Molmo2 pins the very next patch, all three candidates might share 4.57.1 instead of
needing per-candidate versions. Worth testing before assuming permanent version fragmentation.
"""

import re
import time

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    MaxTimeCriteria,
    StoppingCriteriaList,
)

MODEL_ID = "allenai/Molmo2-O-7B"

PROMPT = (
    "Point to every symbol (valve, instrument, flange, nozzle, safety device, or other "
    "P&ID symbol) visible in this image."
)

_POINTS_RE = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')


def load():
    """Returns (processor, model). Requires transformers==4.57.1 (see module docstring)."""
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="auto")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype="auto", device_map="cuda"
    )
    print(f"Loaded {MODEL_ID} in {time.time()-t0:.1f}s")
    print("VRAM used:", f"{torch.cuda.memory_allocated()/1e9:.1f} GB")
    return processor, model


def run(processor, model, image: Image.Image, prompt: str = PROMPT, max_time_s: float = 60.0):
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=2048, do_sample=False,
            stopping_criteria=StoppingCriteriaList([MaxTimeCriteria(max_time=max_time_s)]),
        )
    latency = time.time() - t0
    gen_tokens = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return text, latency


def parse(text, img_w, img_h):
    """Returns (detections, error). Format: <points coords="frame_id x y"/>, values scaled
    by 1000. No native confidence or per-point label documented in this format — fixed
    pseudo-box around the point, confidence None rather than fabricated (see
    common_schema.py for why point-based candidates carry an extra "point" key)."""
    detections = []
    for m in _POINTS_RE.finditer(text):
        nums = [float(v) for v in re.split(r'[\t:;, ]+', m.group(1).strip()) if v]
        if len(nums) % 3 != 0:
            # observed glitch: the first point sometimes has a duplicated leading index,
            # e.g. "1 1 027 016 2 176 158 ..." instead of "1 027 016 2 176 158 ...". If
            # dropping one duplicate near the start fixes the count, repair rather than fail.
            if len(nums) >= 2 and nums[0] == nums[1] and (len(nums) - 1) % 3 == 0:
                nums = nums[1:]
            else:
                return None, f"coords not a multiple of 3 (frame,x,y): {nums}"
        for i in range(0, len(nums), 3):
            _frame, x_scaled, y_scaled = nums[i:i + 3]
            x, y = x_scaled / 1000 * img_w, y_scaled / 1000 * img_h
            detections.append({
                "bbox": [x - 20, y - 20, x + 20, y + 20],
                "confidence": None,
                "entity_type": None,
                "point": [x, y],
            })
    if not detections and ("<point" in text.lower()):
        return None, "contains point-like tags but regex found no matches — check format"
    return detections, None
