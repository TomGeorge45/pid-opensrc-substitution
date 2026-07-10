"""Phase 5.2/5.3 (resequenced, see Stage4_Checklist_Status.md) — combined detection LoRA
fine-tune, trained directly on the raw base rather than a separate domain-adapt-then-LoRA
sequence. Candidate-agnostic: works identically for Qwen3-VL, InternVL3, and Molmo2-O-7B
since all three use the same apply_chat_template + standard forward(labels=...) pattern.

Requires `peft` (not yet installed elsewhere in this repo) — add to the install cell:
    !pip install -q peft

IMPORTANT — timing probe before committing to a full run: per-step training throughput has
never been measured (only rough hand-estimated, in the 17-25h/model ballpark, unmeasured).
Always run `time_training_steps()` on a handful of real steps first and extrapolate from
ACTUAL numbers before launching a long run — do not trust the hand estimate.
"""

import time

from .finetune_dataset import assemble_manifest
from .tiling import tile_and_remap
from .training_format import (
    build_training_example,
    format_boxjson_target,
    format_points_target,
    load_kaggle_class_names,
)


def build_examples(candidate, manifest=None):
    """Returns a flat list of {"image_path", "crop" (None or [x0,y0,x1,y1]), "target_text",
    "n_boxes", "source"} dicts covering Gupta train sheets (tiled per checklist 2.3 — NOT
    raw full sheets, which don't match how the model is actually evaluated) + Kaggle
    pretrain images (already tile-sized, no cropping needed), for one candidate's
    training-target convention."""
    if manifest is None:
        manifest = assemble_manifest()
    class_names = load_kaggle_class_names()

    examples = []
    for item in manifest["gupta_train"]:
        examples.extend(_build_gupta_tiles(item, candidate))
    for item in manifest["kaggle_pretrain"]:
        examples.append(_build_kaggle_example(item, candidate, class_names))
    return examples


def _build_gupta_tiles(item, candidate):
    """Tiles one Gupta sheet per checklist 2.3, produces one training example per tile.
    Gupta is class-agnostic (rule 5) — every box gets entity_type "symbol"."""
    from pathlib import Path

    img, tiles, _orig_boxes = tile_and_remap(Path(item["image_path"]), Path(item["label_path"]))
    examples = []
    for t in tiles:
        tile_w, tile_h = t["x1"] - t["x0"], t["y1"] - t["y0"]
        boxes = [{"bbox": [x0, y0, x1, y1], "entity_type": "symbol"} for _cls, x0, y0, x1, y1 in t["boxes_tile"]]
        target_text = (
            format_boxjson_target(boxes) if candidate in ("qwen3vl", "internvl3")
            else format_points_target(boxes, tile_w, tile_h)
        )
        examples.append({
            "image_path": item["image_path"],
            "crop": [t["x0"], t["y0"], t["x1"], t["y1"]],
            "target_text": target_text,
            "n_boxes": len(boxes),
            "source": "gupta",
        })
    return examples


def _build_kaggle_example(item, candidate, class_names):
    from pathlib import Path

    from PIL import Image

    with Image.open(item["image_path"]) as img:
        w, h = img.size
    ex = build_training_example(
        item["image_path"], Path(item["label_path"]), w, h, "kaggle", candidate, class_names,
    )
    ex["crop"] = None
    ex["source"] = "kaggle"
    return ex


def load_example_image(ex):
    """Opens the example's image, applying its crop region if any (Gupta tiles)."""
    from PIL import Image

    img = Image.open(ex["image_path"]).convert("RGB")
    if ex["crop"] is not None:
        img = img.crop(ex["crop"])
    return img


def build_labeled_inputs(processor, model, image, prompt, target_text, image_first=True):
    """Constructs teacher-forced (input_ids, attention_mask, pixel_values, labels) for one
    example: prompt tokens masked with -100 (not trained on), target_text tokens kept as
    real labels. Works for any candidate using the standard chat-template pattern.

    image_first: match each candidate's own inference message ordering exactly (Qwen3-VL/
    InternVL3 use image-then-text; Molmo2 uses text-then-image — see molmo_candidate.run()).
    A training/inference ordering mismatch is a subtle bug worth avoiding, not a detail."""
    import torch

    image_part = {"type": "image", "image": image}
    text_part = {"type": "text", "text": prompt}
    content = [image_part, text_part] if image_first else [text_part, image_part]

    prompt_messages = [{"role": "user", "content": content}]
    prompt_inputs = processor.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]

    full_messages = prompt_messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": target_text}],
    }]
    full_inputs = processor.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    full_inputs["labels"] = labels
    return {k: v.to(model.device) for k, v in full_inputs.items()}


def setup_lora(model, r=16, alpha=32, dropout=0.05):
    """Wraps model with a LoRA adapter via peft. target_modules="all-linear" avoids needing
    per-architecture module-name lists across 3 different VLM families."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def time_training_steps(processor, model, prompt, examples, n_steps=5, lr=1e-4, image_first=True):
    """Runs n_steps real training steps and reports MEASURED per-step time — the actual
    timing-probe step called out in this module's docstring. Never skip this before a full
    run; the hand estimate in Stage4_Checklist_Status.md is explicitly not to be trusted."""
    import torch

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )

    step_times = []
    for i, ex in enumerate(examples[:n_steps]):
        t0 = time.time()
        img = load_example_image(ex)
        inputs = build_labeled_inputs(processor, model, img, prompt, ex["target_text"], image_first=image_first)
        out = model(**inputs)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_time = time.time() - t0
        step_times.append(step_time)
        print(f"step {i+1}/{n_steps}: loss={loss.item():.4f} time={step_time:.2f}s")

    avg_step_time = sum(step_times) / len(step_times)
    print(f"\nMeasured avg step time: {avg_step_time:.2f}s")
    return avg_step_time
