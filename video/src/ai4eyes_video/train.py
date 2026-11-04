"""Supervised training entry point: trains the ActionFormer-style head on
top of a frozen V-JEPA2.1 backbone for one test type (NITBUT /
INVASIVE_TBUT / BLINK_MECHANICS).

Usage:
    python -m ai4eyes_video.train --config video/configs/supervised/nitbut.yaml

Resume: pass --resume (or set RESUME=1 in the environment, which
scripts/train_supervised.sh forwards) to continue from
<output_dir>/last.pt instead of starting fresh.
"""
import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .annotations import read_annotations
from .backbone import VJEPA21Backbone
from .config import paths
from .datasets import ClipTrainDataset, train_collate
from .model import TemporalActionDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a configs/supervised/<test>.yaml file")
    parser.add_argument("--output-dir", default=None, help="Overrides paths().checkpoint_dir/<test_type>")
    parser.add_argument(
        "--resume", action="store_true", default=os.environ.get("RESUME", "0") == "1",
        help="Resume from <output_dir>/last.pt instead of starting fresh (also triggered by RESUME=1)",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    p = paths()

    test_type = cfg["test_type"]
    classes = cfg["classes"]
    output_dir = Path(args.output_dir or Path(p.checkpoint_dir) / test_type.lower())
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = cfg.get("annotations_path") or str(Path(p.annotations_dir) / cfg["annotations_filename"])
    gt = read_annotations(annotations_path)
    train_records = gt.split("train")
    if not train_records:
        raise RuntimeError(f"No training records in {annotations_path} (split='train')")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_cfg = cfg["backbone"]
    backbone = VJEPA21Backbone(
        repo_path=backbone_cfg.get("repo_path", p.vjepa2_repo),
        checkpoint_path=backbone_cfg.get("checkpoint_path", p.vjepa2_checkpoint),
        model_name=backbone_cfg.get("model_name", "vjepa2_1_vit_large_384"),
        checkpoint_key=backbone_cfg.get("checkpoint_key", "ema_encoder"),
        num_frames=backbone_cfg.get("num_frames", 16),
        patch_size=backbone_cfg.get("patch_size", 16),
        tubelet_size=backbone_cfg.get("tubelet_size", 2),
        freeze=backbone_cfg.get("freeze", True),
    ).to(device)

    window_frames = cfg.get("window_frames", 160)
    resolution = cfg.get("resolution", 224)
    model = TemporalActionDetector(
        backbone=backbone,
        num_classes=len(classes),
        window_frames=window_frames,
        head_channels=cfg.get("head", {}).get("head_channels", 512),
    ).to(device)

    dataset = ClipTrainDataset(
        records=train_records, classes=classes, video_root=p.video_root,
        window_frames=window_frames, resolution=resolution,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.get("batch_size", 2), shuffle=True,
        num_workers=cfg.get("num_workers", 4), collate_fn=train_collate, drop_last=True,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 0.05))
    )
    epochs = cfg.get("epochs", 60)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(loader)))

    start_epoch = 0
    last_ckpt_path = output_dir / "last.pt"
    if args.resume and last_ckpt_path.is_file():
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {last_ckpt_path} at epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        for step, batch in enumerate(loader):
            frames = batch["frames"].to(device)
            gt_segments = [s.to(device) for s in batch["gt_segments"]]
            gt_labels = [l.to(device) for l in batch["gt_labels"]]

            losses = model.forward_train(frames, gt_segments, gt_labels)
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += losses["loss"].item()

            if step % cfg.get("log_interval", 20) == 0:
                print(
                    f"epoch {epoch} step {step}/{len(loader)} "
                    f"loss={losses['loss'].item():.4f} cls={losses['cls_loss'].item():.4f} reg={losses['reg_loss'].item():.4f}"
                )

        print(f"epoch {epoch} done, mean loss={running_loss / max(1, len(loader)):.4f}")
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "config": cfg,
            },
            last_ckpt_path,
        )
        if (epoch + 1) % cfg.get("checkpoint_interval", 5) == 0 or epoch + 1 == epochs:
            torch.save(model.state_dict(), output_dir / f"checkpoint-{epoch}.pt")

    # `last.pt` (above) is a resumable checkpoint: {"model", "optimizer",
    # "scheduler", "epoch", "config"}. `final.pt` is a bare model
    # state_dict -- the format evaluate.py/time_inference.py load directly,
    # named so callers never have to guess the final epoch number.
    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Resumable checkpoint: {last_ckpt_path}. For evaluate.py/time_inference.py: {final_path}")


if __name__ == "__main__":
    main()
