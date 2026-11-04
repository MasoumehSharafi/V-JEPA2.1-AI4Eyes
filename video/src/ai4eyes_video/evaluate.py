"""Evaluation entry point, split into two explicit phases so metrics are
ALWAYS computed from a persisted prediction file, never inline during
inference or training:

  1. inference: run the checkpoint over the sliding-window test clips,
     stitch window-local predictions into video-level (seconds) detections,
     and write them to --predictions-out as JSON.
  2. score: load --predictions-out + the ground-truth annotations and
     compute metrics.py's functions, printing a report (and optionally
     writing --metrics-out).

Usage:
    python -m ai4eyes_video.evaluate --config video/configs/supervised/nitbut.yaml \\
        --checkpoint checkpoints/nitbut/checkpoint-59.pt --split test \\
        --predictions-out outputs/nitbut_test_predictions.json \\
        --metrics-out outputs/nitbut_test_metrics.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .annotations import read_annotations
from .backbone import VJEPA21Backbone
from .config import paths
from .datasets import ClipSlidingDataset
from .heads import nms_1d
from .metrics import count_metrics, gradability_metrics
from .model import TemporalActionDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="A model state_dict .pt file (see train.py's checkpoint-N.pt)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--predictions-out", required=True, help="Where to write the persisted prediction JSON")
    parser.add_argument("--metrics-out", default=None, help="Optional: also write the metrics report as JSON")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5, help="tIoU threshold for scoring (count_metrics only)")
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Skip phase 1 and score an already-written --predictions-out file (e.g. to re-score at a different threshold)",
    )
    return parser


def run_inference(cfg: dict, checkpoint_path: str, split: str, score_threshold: float, nms_iou: float) -> dict:
    p = paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotations_path = cfg.get("annotations_path") or str(Path(p.annotations_dir) / cfg["annotations_filename"])
    gt = read_annotations(annotations_path)
    records = gt.split(split)
    if not records:
        raise RuntimeError(f"No records in {annotations_path} for split={split!r}")

    backbone_cfg = cfg["backbone"]
    backbone = VJEPA21Backbone(
        repo_path=backbone_cfg.get("repo_path", p.vjepa2_repo),
        checkpoint_path=backbone_cfg.get("checkpoint_path", p.vjepa2_checkpoint),
        model_name=backbone_cfg.get("model_name", "vjepa2_1_vit_large_384"),
        checkpoint_key=backbone_cfg.get("checkpoint_key", "ema_encoder"),
        num_frames=backbone_cfg.get("num_frames", 16),
        patch_size=backbone_cfg.get("patch_size", 16),
        tubelet_size=backbone_cfg.get("tubelet_size", 2),
        freeze=True,
    ).to(device)

    window_frames = cfg.get("window_frames", 160)
    model = TemporalActionDetector(
        backbone=backbone, num_classes=len(cfg["classes"]), window_frames=window_frames,
        head_channels=cfg.get("head", {}).get("head_channels", 512),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = ClipSlidingDataset(
        records=records, video_root=p.video_root, window_frames=window_frames,
        overlap_ratio=cfg.get("eval_overlap_ratio", 0.5), resolution=cfg.get("resolution", 224),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 4),
                         collate_fn=lambda batch: batch[0])

    raw_detections = defaultdict(list)
    for window in loader:
        frames = window.frames.unsqueeze(0).to(device)
        detections = model.forward_test(frames, score_threshold=score_threshold, nms_iou=nms_iou)[0]
        for det in detections:
            start_f = window.window_start_frame + det["segment"][0]
            end_f = window.window_start_frame + det["segment"][1]
            raw_detections[window.key].append(
                {"segment": [start_f / window.fps, end_f / window.fps], "label": det["label"], "score": det["score"]}
            )

    predictions = {}
    for key, dets in raw_detections.items():
        merged_by_class = defaultdict(list)
        for d in dets:
            merged_by_class[d["label"]].append(d)
        final = []
        for label, group in merged_by_class.items():
            segs = torch.tensor([g["segment"] for g in group], dtype=torch.float32)
            scores = torch.tensor([g["score"] for g in group], dtype=torch.float32)
            keep = nms_1d(segs, scores, iou_threshold=nms_iou)
            for i in keep.tolist():
                final.append({"segment": segs[i].tolist(), "label": label, "score": scores[i].item()})
        predictions[key] = final
    return predictions


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())

    predictions_path = Path(args.predictions_out)
    if args.skip_inference:
        predictions = json.loads(predictions_path.read_text())
    else:
        predictions = run_inference(cfg, args.checkpoint, args.split, args.score_threshold, args.nms_iou)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(json.dumps(predictions, indent=2))
        print(f"Wrote predictions for {len(predictions)} videos to {predictions_path}")

    p = paths()
    annotations_path = cfg.get("annotations_path") or str(Path(p.annotations_dir) / cfg["annotations_filename"])
    gt = read_annotations(annotations_path)

    if len(cfg["classes"]) == 1:
        report = gradability_metrics(gt, predictions, args.split, score_threshold=args.score_threshold)
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    else:
        report = count_metrics(gt, predictions, args.split, score_threshold=args.score_threshold, iou_threshold=args.match_iou)
        print(json.dumps(report, indent=2))

    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in report.items() if k != "rows"}
        out.write_text(json.dumps(serializable, indent=2))
        print(f"Wrote metrics to {out}")


if __name__ == "__main__":
    main()
