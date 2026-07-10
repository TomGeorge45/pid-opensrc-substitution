"""Phase 5.1/5.2 groundwork — convert ground-truth YOLO labels into each candidate's own
training-target format (the same convention it's parsed/scored with at inference time —
qwen_candidate.parse / molmo_candidate.parse), so training targets and eval predictions are
never subtly inconsistent with each other.

Gupta boxes are class-agnostic ("Symbol" — checklist rule 5), so they always get entity_type
"symbol" regardless of candidate. Kaggle boxes get their real class name from classes.json
(built in Stage4_Phase2_Data_Preparation.ipynb, checklist 2.2) when available.
"""

import json

from .data_utils import ROOT

GUPTA_GENERIC_ENTITY_TYPE = "symbol"


def _yolo_line_to_xyxy(line, img_w, img_h):
    parts = line.split()
    cls_id = parts[0]
    cx, cy, w, h = (float(v) for v in parts[1:5])
    cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    return cls_id, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def load_kaggle_class_names():
    """Returns {class_id_str: kaggle_name} from classes.json, or {} if not present."""
    classes_path = ROOT / "classes.json"
    if not classes_path.exists():
        return {}
    data = json.loads(classes_path.read_text())
    return {cid: entry["kaggle_name"] for cid, entry in data["classes"].items()}


def boxes_from_label_file(label_path, img_w, img_h, source, class_names=None):
    """source: "gupta" (class-agnostic) or "kaggle" (typed, needs class_names map).
    Returns list of {"bbox": [x0,y0,x1,y1], "entity_type": str}."""
    class_names = class_names or {}
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls_id, x0, y0, x1, y1 = _yolo_line_to_xyxy(line, img_w, img_h)
        if source == "gupta":
            entity_type = GUPTA_GENERIC_ENTITY_TYPE
        else:
            entity_type = class_names.get(cls_id, f"class_{cls_id}")
        boxes.append({"bbox": [x0, y0, x1, y1], "entity_type": entity_type})
    return boxes


def format_boxjson_target(boxes):
    """Target text for Qwen3-VL / InternVL3 — matches qwen_candidate's array-of-arrays
    format exactly: [x0, y0, x1, y1, confidence, "entity_type"]. Ground truth confidence
    is always 1.0 (it's the correct answer, not a model guess)."""
    rows = [
        [round(b["bbox"][0]), round(b["bbox"][1]), round(b["bbox"][2]), round(b["bbox"][3]), 1.0, b["entity_type"]]
        for b in boxes
    ]
    return json.dumps(rows)


def format_points_target(boxes, img_w, img_h):
    """Target text for Molmo2 — matches molmo_candidate's <points coords="frame x y"/>
    format, one point per box (box center), coordinates scaled 0-1000 per the real format."""
    if not boxes:
        # the parser's coords regex requires >=1 char (matches real model output, which
        # never emits an empty coords="" attribute either) — so "no boxes" is an empty
        # string, not a <points> tag with nothing in it.
        return ""
    parts = []
    for i, b in enumerate(boxes, start=1):
        x0, y0, x1, y1 = b["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x_scaled = round(cx / img_w * 1000)
        y_scaled = round(cy / img_h * 1000)
        parts.append(f"{i} {x_scaled} {y_scaled}")
    coords = " ".join(parts)
    return f'<points coords="{coords}"></points>'


def build_training_example(image_path, label_path, img_w, img_h, source, candidate, class_names=None):
    """Returns {"image_path": str, "target_text": str} for the given candidate
    ("qwen3vl" | "internvl3" | "molmo2")."""
    boxes = boxes_from_label_file(label_path, img_w, img_h, source, class_names)
    if candidate in ("qwen3vl", "internvl3"):
        target_text = format_boxjson_target(boxes)
    elif candidate == "molmo2":
        target_text = format_points_target(boxes, img_w, img_h)
    else:
        raise ValueError(f"unknown candidate: {candidate}")
    return {"image_path": str(image_path), "target_text": target_text, "n_boxes": len(boxes)}
