"""InternVL3-8B candidate: load, prompt, parse.

Reuses Qwen's prompt/parser as-is (checklist 2.5) — no model-specific tuning was needed to
get a working first result; InternVL3 cooperated with the same array-of-arrays format.
"""

import time
import typing

if typing.TYPE_CHECKING:
    from PIL import Image

from .qwen_candidate import SYMBOL_DETECTION_PROMPT, parse as parse_qwen_output

MODEL_ID = "OpenGVLab/InternVL3-8B-hf"

# re-exported so callers can `from internvl_candidate import parse` symmetrically with the
# other candidate modules
parse = parse_qwen_output


def load():
    """Returns (processor, model). ~16GB VRAM.
    Imports torch/transformers lazily so this module can be imported (and `parse()` tested)
    on machines without them installed — only load()/run() actually need the GPU stack."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"Loaded {MODEL_ID} in {time.time()-t0:.1f}s")
    print("VRAM used:", f"{torch.cuda.memory_allocated()/1e9:.1f} GB")
    return processor, model


def run(processor, model, image: "Image.Image", prompt: str = SYMBOL_DETECTION_PROMPT, max_time_s: float = 60.0):
    import torch
    from transformers import MaxTimeCriteria, StoppingCriteriaList

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
