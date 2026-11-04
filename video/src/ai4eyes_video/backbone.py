"""Bridge from the cloned facebookresearch/vjepa2 checkout to a plain
torch.nn.Module feature extractor. This is the ONLY place that imports
from the vjepa2 repo -- everything downstream (heads.py, model.py) just
consumes a [B, C, T', H', W'] feature map and never knows vjepa2 exists.

The encoder is frozen by default: every result in ../README.md used a
fully frozen backbone (zero gradient to V-JEPA weights), with only the
detection head (heads.py) trained.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn


class VJEPA21Backbone(nn.Module):
    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        model_name: str = "vjepa2_1_vit_large_384",
        checkpoint_key: str = "ema_encoder",
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        freeze: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.freeze_encoder = freeze

        repo_path = str(Path(repo_path).resolve())
        if not Path(repo_path).is_dir():
            raise FileNotFoundError(
                f"V-JEPA2 repository not found at {repo_path!r} -- run scripts/setup_vjepa.sh first."
            )
        checkpoint_path = str(Path(checkpoint_path).resolve())
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"V-JEPA2 checkpoint not found at {checkpoint_path!r}.")

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from src.hub import backbones as vjepa_hub_backbones  # noqa: PLC0415 (lazy, repo-path-dependent import)

        if not hasattr(vjepa_hub_backbones, model_name):
            raise ValueError(
                f"Unknown V-JEPA2 model_name {model_name!r}; "
                f"expected one of the vjepa2_1_vit_*_384 hub constructors in {repo_path}/src/hub.py."
            )
        model_fn = getattr(vjepa_hub_backbones, model_name)
        encoder, predictor = model_fn(pretrained=False, num_frames=num_frames, tubelet_size=tubelet_size)
        del predictor  # the SSL predictor head is not needed for downstream detection

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_key not in checkpoint:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no {checkpoint_key!r} key. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        state_dict = {}
        for key, value in checkpoint[checkpoint_key].items():
            key = key.replace("module.", "").replace("backbone.", "")
            state_dict[key] = value
        message = encoder.load_state_dict(state_dict, strict=True)
        print(f"[VJEPA21Backbone] loaded {checkpoint_key!r} from {checkpoint_path}: {message}")

        self.encoder = encoder
        self.embed_dim = encoder.embed_dim

        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True) -> "VJEPA21Backbone":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # keep BatchNorm/Dropout frozen in eval mode even if the outer module trains
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, T, H, W] -> [B, embed_dim, T/tubelet_size, H/patch_size, W/patch_size]."""
        if x.ndim != 5:
            raise ValueError(f"Expected [B,3,T,H,W], got shape {tuple(x.shape)}")
        batch, _, frames, height, width = x.shape
        if frames % self.tubelet_size != 0:
            raise ValueError(f"frames={frames} must be divisible by tubelet_size={self.tubelet_size}")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"H,W=({height},{width}) must be divisible by patch_size={self.patch_size}")

        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            tokens = self.encoder(x)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected encoder output [B,N,C], got shape {tuple(tokens.shape)}")

        t_tok = frames // self.tubelet_size
        h_tok = height // self.patch_size
        w_tok = width // self.patch_size
        expected = t_tok * h_tok * w_tok
        if tokens.shape[1] != expected:
            raise RuntimeError(f"Expected {expected} tokens ({t_tok}x{h_tok}x{w_tok}), got {tokens.shape[1]}")

        features = tokens.reshape(batch, t_tok, h_tok, w_tok, self.embed_dim)
        features = features.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,C] -> [B,C,T,H,W]
        return features
