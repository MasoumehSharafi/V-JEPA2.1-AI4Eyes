"""Inference-latency timing: model load time, then per-eye read / infer /
post-process time, broken out separately so I/O cost isn't mistaken for
model cost. Does not train or modify the checkpoint.

Usage:
    python -m ai4eyes_video.time_inference --config video/configs/supervised/nitbut.yaml \\
        --checkpoint checkpoints/nitbut/checkpoint-59.pt --split test \\
        --max-videos 10 --report-out outputs/timing/nitbut_timing.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from .annotations import read_annotations
from .backbone import VJEPA21Backbone
from .config import paths
from .datasets import ClipSlidingDataset
from .model import TemporalActionDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--report-out", default=None)
    return parser


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(round(pct / 100 * (len(values) - 1))))
    return values[idx]


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    p = paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotations_path = cfg.get("annotations_path") or str(Path(p.annotations_dir) / cfg["annotations_filename"])
    gt = read_annotations(annotations_path)
    records = gt.split(args.split)[: args.max_videos]
    if not records:
        raise RuntimeError(f"No records in {annotations_path} for split={args.split!r}")

    load_start = time.perf_counter()
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    load_time_s = time.perf_counter() - load_start

    dataset = ClipSlidingDataset(
        records=records, video_root=p.video_root, window_frames=window_frames,
        overlap_ratio=cfg.get("eval_overlap_ratio", 0.5), resolution=cfg.get("resolution", 224),
    )

    read_times, infer_times, post_times = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            t0 = time.perf_counter()
            window = dataset[i]
            t1 = time.perf_counter()
            frames = window.frames.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            _ = model.forward_test(frames)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            read_times.append(t1 - t0)
            infer_times.append(t3 - t2)
            post_times.append(0.0)  # NMS is already included inside forward_test's timed span above

    def summarize(values: list[float]) -> dict:
        return {
            "median_s": statistics.median(values), "p10_s": percentile(values, 10),
            "p90_s": percentile(values, 90), "mean_s": statistics.mean(values), "n": len(values),
        }

    report = {
        "test_type": cfg["test_type"],
        "device": str(device),
        "model_load_s": load_time_s,
        "n_windows": len(dataset),
        "n_videos": len(records),
        "read_s": summarize(read_times),
        "infer_s": summarize(infer_times),
        "total_s_per_window": summarize([r + i for r, i in zip(read_times, infer_times)]),
    }
    print(json.dumps(report, indent=2))
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Wrote timing report to {out}")


if __name__ == "__main__":
    main()
