"""Metric functions. These are PURE functions over already-persisted
ground truth (an AnnotationSet) and predictions (a plain dict loaded from
a JSON file written by evaluate.py) -- never called from inside a training
loop. evaluate.py is the only place that wires a checkpoint to these.
"""
from statistics import mean

from .annotations import AnnotationSet


def temporal_iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def gradability_metrics(
    gt: AnnotationSet,
    predictions: dict[str, list[dict]],
    split: str,
    score_threshold: float = 0.1,
) -> dict:
    """For single-instance-per-video tasks (NITBUT, Invasive-TBUT): is
    there a gradable event, and if so, how accurate is its timing. A video
    is "predicted gradable" if its best-scoring detection >= score_threshold."""
    rows = []
    for record in gt.split(split):
        gt_gradable = len(record.annotations) > 0
        gt_segment = record.annotations[0]["segment"] if gt_gradable else None

        preds = predictions.get(record.key, [])
        best = max(preds, key=lambda p: p["score"], default=None)
        pred_gradable = best is not None and best["score"] >= score_threshold

        row = {"key": record.key, "gt_gradable": gt_gradable, "pred_gradable": pred_gradable}
        if gt_gradable and pred_gradable:
            row["tiou"] = temporal_iou(gt_segment, best["segment"])
            row["start_error"] = abs(best["segment"][0] - gt_segment[0])
            row["end_error"] = abs(best["segment"][1] - gt_segment[1])
            row["duration_error"] = abs(
                (best["segment"][1] - best["segment"][0]) - (gt_segment[1] - gt_segment[0])
            )
        rows.append(row)

    tp = sum(1 for r in rows if r["gt_gradable"] and r["pred_gradable"])
    fn = sum(1 for r in rows if r["gt_gradable"] and not r["pred_gradable"])
    fp = sum(1 for r in rows if not r["gt_gradable"] and r["pred_gradable"])
    tn = sum(1 for r in rows if not r["gt_gradable"] and not r["pred_gradable"])
    n = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    result = {
        "n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": precision, "recall": recall, "f1": f1,
        "rows": rows,
    }
    tp_rows = [r for r in rows if "tiou" in r]
    if tp_rows:
        result["mean_tiou"] = mean(r["tiou"] for r in tp_rows)
        result["start_mae"] = mean(r["start_error"] for r in tp_rows)
        result["end_mae"] = mean(r["end_error"] for r in tp_rows)
        result["duration_mae"] = mean(r["duration_error"] for r in tp_rows)
    return result


def _match_instances(gt_segments: list[list[float]], preds: list[dict], iou_threshold: float) -> tuple[int, int, int]:
    """Greedy highest-score-first matching. Returns (tp, fp, fn)."""
    unmatched = list(range(len(gt_segments)))
    tp = fp = 0
    for pred in sorted(preds, key=lambda p: -p["score"]):
        best_idx, best_iou = None, 0.0
        for idx in unmatched:
            iou = temporal_iou(pred["segment"], gt_segments[idx])
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            unmatched.remove(best_idx)
        else:
            fp += 1
    return tp, fp, len(unmatched)


def count_metrics(
    gt: AnnotationSet,
    predictions: dict[str, list[dict]],
    split: str,
    score_threshold: float = 0.1,
    iou_threshold: float = 0.5,
) -> dict:
    """For multi-instance, multi-class tasks (Blink Mechanics): per-class
    detection matching (TP/FP/FN -> Precision/Recall/F1) plus per-video
    count-exact-match (predicted instance count == ground-truth count)."""
    classes = gt.classes
    totals = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    count_correct = {c: 0 for c in classes}
    n = 0

    for record in gt.split(split):
        n += 1
        gt_by_class = {c: [] for c in classes}
        for ann in record.annotations:
            gt_by_class[ann["label"]].append(ann["segment"])

        preds = [p for p in predictions.get(record.key, []) if p["score"] >= score_threshold]
        preds_by_class = {c: [] for c in classes}
        for p in preds:
            label = classes[p["label"]] if isinstance(p["label"], int) else p["label"]
            if label in preds_by_class:
                preds_by_class[label].append(p)

        for c in classes:
            tp, fp, fn = _match_instances(gt_by_class[c], preds_by_class[c], iou_threshold)
            totals[c]["tp"] += tp
            totals[c]["fp"] += fp
            totals[c]["fn"] += fn
            if len(gt_by_class[c]) == len(preds_by_class[c]):
                count_correct[c] += 1

    per_class = {}
    for c in classes:
        t = totals[c]
        precision = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) else float("nan")
        recall = t["tp"] / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        per_class[c] = {
            **t, "precision": precision, "recall": recall, "f1": f1,
            "count_exact_match": count_correct[c] / n if n else float("nan"),
        }
    return {"n": n, "per_class": per_class}
