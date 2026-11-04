"""Extracts just the encoder sub-state-dict from a full V-JEPA2.1 SSL
training checkpoint (which also contains the predictor, optimizer state,
etc.) into a small file that backbone.VJEPA21Backbone can load directly.

Usage:
    python -m ai4eyes_video.export_encoder \\
        --checkpoint work/vjepa2_ssl/ai4eyes_allsigns/best.pth.tar \\
        --key ema_encoder \\
        --out checkpoints/vjepa_dapt_encoder.pt
"""
import argparse
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Full V-JEPA2.1 training checkpoint (.pth.tar)")
    parser.add_argument("--key", default="ema_encoder", help="Which sub-state-dict to extract (default: ema_encoder)")
    parser.add_argument("--out", required=True, help="Output path for the extracted encoder-only checkpoint")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.key not in checkpoint:
        raise KeyError(f"{args.checkpoint} has no {args.key!r} key. Available keys: {list(checkpoint.keys())}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({args.key: checkpoint[args.key]}, out_path)
    n_params = sum(t.numel() for t in checkpoint[args.key].values() if hasattr(t, "numel"))
    print(f"Extracted {args.key!r} ({n_params:,} parameters) from {args.checkpoint} -> {out_path}")


if __name__ == "__main__":
    main()
