"""Phase 3 metric harness — detection (Gupta) and typing (Kaggle) scoring.

Decisions locked (2026-07-10), see Stage4_Checklist_Status.md for full rationale:
  - 3.2 match rule: point-in-GT-box hit. Box predictions use their center point; point
    predictions (Molmo2) use the point directly. Fair to both — no artificial box-size
    parameter invented for Molmo, no IoU needed for a model that never predicted a box.
  - 3.1 Part B scope: typing is scored on GROUND-TRUTH CROPS (localization handed to the
    model for free) — isolates pure typing ability, not entangled with detection errors.
  - 3.6 pass bar (provisional, pending the deferred Claude Stage-4 incumbent baseline run):
      detection (Gupta): mAP@0.5 >= 0.70 OR F1 >= 0.70
      typing (Kaggle, GT-crop mode): accuracy >= 0.80
      fallback trigger ("[X]% of Claude's Stage-4 recall", spec section 1): TODO, blocked
      on the incumbent baseline run (explicitly deferred, not done here).

Never average detection and typing into one number (CLAUDE.md hard rule 5) — these live as
two separate function families for that reason, not because it's more code.
"""

from collections import Counter

# --- 3.5 fixed constants ---
IOU_THRESHOLD = 0.5          # documented for box-format analyses; NOT the primary match rule
BOX_FORMAT = "xyxy_absolute"  # matches common_schema.py's bbox convention
RARE_CLASS_MAX_INSTANCES = 20  # per checklist 3.1: classes with <20 instances are "rare"

# --- 3.6 provisional pass bar ---
DETECTION_MAP50_BAR = 0.70
DETECTION_F1_BAR = 0.70
TYPING_ACCURACY_BAR = 0.80
FALLBACK_TRIGGER_PCT_OF_CLAUDE_RECALL = None  # TODO — blocked on deferred incumbent baseline


def _bbox_center(bbox):
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def point_in_box(point, box):
    """point: (x, y). box: [x0, y0, x1, y1]. Returns bool."""
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def prediction_point(pred):
    """Unified match-point for any prediction, box or point-based (3.2 decision)."""
    if "point" in pred and pred["point"] is not None:
        return tuple(pred["point"])
    return _bbox_center(pred["bbox"])


def match_predictions_to_gt(predictions, gt_boxes, require_type=None):
    """Greedy one-to-one matching: each GT box matched by at most one prediction, each
    prediction matches at most one GT box (prevents one prediction claiming credit for many
    GT boxes it happens to overlap). Predictions processed in confidence-descending order
    when confidence is available (None sorts last).

    predictions: list of common-schema dicts.
    gt_boxes: list of [x0, y0, x1, y1].
    require_type: if given, an entity_type string — only count a match if
        pred['entity_type'] == require_type (used for typing eval, not detection).

    Returns (n_matched, n_predictions, n_gt).
    """
    order = sorted(
        range(len(predictions)),
        key=lambda i: (predictions[i].get("confidence") is None, -(predictions[i].get("confidence") or 0)),
    )
    gt_taken = [False] * len(gt_boxes)
    n_matched = 0
    for i in order:
        pred = predictions[i]
        if require_type is not None and pred.get("entity_type") != require_type:
            continue
        pt = prediction_point(pred)
        for j, box in enumerate(gt_boxes):
            if not gt_taken[j] and point_in_box(pt, box):
                gt_taken[j] = True
                n_matched += 1
                break
    return n_matched, len(predictions), len(gt_boxes)


def precision_recall_f1(predictions, gt_boxes):
    """Class-agnostic detection metrics (Gupta — 'is a symbol here', not what type).
    Confirm bar (3.1): perfect prediction -> 1.0, empty -> 0.0, half-correct -> sane middle.
    """
    n_matched, n_pred, n_gt = match_predictions_to_gt(predictions, gt_boxes)
    precision = n_matched / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = n_matched / n_gt if n_gt else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_matched": n_matched, "n_pred": n_pred, "n_gt": n_gt}


def average_precision(predictions, gt_boxes):
    """Confidence-ranked AP — requires predictions with real (non-None) confidence values.
    Returns None if no predictions have confidence (e.g. Molmo2 — see common_schema.py)."""
    scored = [p for p in predictions if p.get("confidence") is not None]
    if not scored or not gt_boxes:
        return None
    scored = sorted(scored, key=lambda p: -p["confidence"])
    gt_taken = [False] * len(gt_boxes)
    precisions, recalls = [], []
    n_matched = 0
    for i, pred in enumerate(scored, start=1):
        pt = prediction_point(pred)
        hit = False
        for j, box in enumerate(gt_boxes):
            if not gt_taken[j] and point_in_box(pt, box):
                gt_taken[j] = True
                hit = True
                break
        if hit:
            n_matched += 1
        precisions.append(n_matched / i)
        recalls.append(n_matched / len(gt_boxes))

    # standard AP: area under the monotone-decreasing precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap


def typing_accuracy_gt_crop(predicted_types, true_types):
    """Part B, GT-crop mode: predicted_types[i] is the model's classification of the i-th
    ground-truth crop; true_types[i] is its real Kaggle class name. Localization is not
    scored here — every crop is a real symbol by construction.

    Returns {"accuracy": ..., "per_class": {...}, "rare_class_recall": {...}}.
    """
    assert len(predicted_types) == len(true_types)
    n = len(true_types)
    if n == 0:
        return {"accuracy": 0.0, "per_class": {}, "rare_class_recall": {}}

    correct = sum(p == t for p, t in zip(predicted_types, true_types))
    accuracy = correct / n

    class_totals = Counter(true_types)
    class_correct = Counter(t for p, t in zip(predicted_types, true_types) if p == t)
    per_class = {cls: class_correct[cls] / class_totals[cls] for cls in class_totals}

    rare_classes = {cls for cls, count in class_totals.items() if count < RARE_CLASS_MAX_INSTANCES}
    rare_class_recall = {cls: per_class[cls] for cls in rare_classes}

    return {"accuracy": accuracy, "per_class": per_class, "rare_class_recall": rare_class_recall}
