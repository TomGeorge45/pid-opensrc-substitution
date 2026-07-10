"""Qwen3-VL-8B-Instruct candidate: load, prompt, parse.

Prompt uses a compact array-of-arrays format rather than array-of-objects — an earlier
object-with-keys format caused repeated JSON field names ("confidence", "entity_type") to
trigger degenerate generation loops when combined with repetition-penalty settings; the
key-name-free format avoids the problem at the source. See Stage4_Checklist_Status.md item 2.5
for the full incident history.
"""

import json
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

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SYMBOL_DETECTION_PROMPT = """You are analyzing a tile cropped from a Piping & Instrumentation \
Diagram (P&ID). Detect every symbol (valve, instrument, flange, nozzle, safety device, or \
other P&ID symbol) visible in this image.

Respond with ONLY a JSON array of arrays, no other text, no explanation. Each inner array is
exactly: [x0, y0, x1, y1, confidence, "entity_type"]

Example: [[100, 200, 150, 260, 0.95, "valve"], [400, 50, 430, 90, 0.88, "instrument"]]

Coordinates are absolute pixel coordinates in this image (top-left origin, x right, y down),
NOT normalized. confidence is a float 0.0-1.0. If you see no symbols, respond with an empty
array: []"""


def load():
    """Returns (processor, model). ~17.5GB VRAM on bfloat16."""
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"Loaded {MODEL_ID} in {time.time()-t0:.1f}s")
    print("VRAM used:", f"{torch.cuda.memory_allocated()/1e9:.1f} GB")
    return processor, model


def run(processor, model, image: Image.Image, prompt: str = SYMBOL_DETECTION_PROMPT, max_time_s: float = 60.0):
    """max_time_s is a hard wall-clock cap — protects against degenerate generation loops that
    would otherwise burn the full max_new_tokens budget (observed: a stuck loop can take 170s+
    on a dense tile). Distinct from max_new_tokens, which only caps token COUNT, not time."""
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=3072, do_sample=False,
            stopping_criteria=StoppingCriteriaList([MaxTimeCriteria(max_time=max_time_s)]),
        )
    latency = time.time() - t0
    gen_tokens = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return text, latency


def parse(text):
    """Returns (detections, error). detections is a list of common-schema dicts, or None if
    parsing failed. error is None on success, else a short description."""
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    # repair a real observed glitch: model emits "0:97" instead of "0.97" (colon for decimal point)
    cleaned = re.sub(r'(?<=: )(\d):(\d+)(?=[,}\s])', r'\1.\2', cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    if not isinstance(data, list):
        return None, f"expected a JSON array, got {type(data).__name__}"
    detections = []
    for i, item in enumerate(data):
        if not (isinstance(item, list) and len(item) == 6):
            return None, f"item {i} malformed, expected [x0,y0,x1,y1,confidence,entity_type]: {item}"
        x0, y0, x1, y1, confidence, entity_type = item
        detections.append({
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "confidence": float(confidence),
            "entity_type": str(entity_type),
        })
    return detections, None
