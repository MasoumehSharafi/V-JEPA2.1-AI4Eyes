"""V-JEPA 2.1 ViT-L/16 encoder on still images (code: facebookresearch/vjepa2, MIT; cloned by scripts/setup_vjepa.sh).

An image goes through the encoder's image path (1-frame patch embedding + image modality embedding); the 3D RoPE
positions are interpolated, so any square resolution that is a multiple of 16 works (448 for the signs, 384 for SSL).
"""
import sys

import torch
import torch.nn as nn

from ai4eyes_image import paths

PATCH = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RELEASED_ENCODER = "vjepa2/vjepa2_1_vitl_384_encoder.pt"      # under $AI4EYES_CHECKPOINT_DIR
RELEASED_PREDICTOR = "vjepa2/vjepa2_1_vitl_384_predictor.pt"


def vjepa_modules():
    """(vision_transformer, predictor) modules of the official V-JEPA 2.1 code."""
    root = str(paths.VJEPA2_DIR)
    if not (paths.VJEPA2_DIR / "app" / "vjepa_2_1").is_dir():
        raise SystemExit(f"V-JEPA 2 code not found at {root}; run scripts/setup_vjepa.sh")
    if root not in sys.path:
        sys.path.insert(0, root)
    from app.vjepa_2_1.models import predictor, vision_transformer
    return vision_transformer, predictor


def build_encoder(res=384):
    vt, _ = vjepa_modules()
    return vt.vit_large(patch_size=PATCH, img_size=(res, res), num_frames=64, tubelet_size=2, use_sdpa=True,
                        use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                        img_temporal_dim_size=1, interpolate_rope=True)


class FrozenVJEPA(nn.Module):
    """Frozen encoder. Input [B,3,H,W] ImageNet-normalised → final patch tokens after the last norm [B,(H/16)^2,1024].
    No CLS token, no registers. embed() / call_block() / final() are the pieces the LoRA model runs through."""

    patch_size = PATCH

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk
        self.embed_dim = trunk.embed_dim
        self._grid = None
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        assert x.ndim == 4 and x.shape[-1] == x.shape[-2] and x.shape[-1] % self.patch_size == 0, x.shape
        x = self.embed(x)
        for blk in self.trunk.blocks:
            x = self.call_block(blk, x)
        return self.final(x)

    def embed(self, x):
        self._grid = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        x = self.trunk.patch_embed_img(x.unsqueeze(2))           # [B,3,1,H,W] → image patch embedding
        if self.trunk.modality_embedding:
            x = x + self.trunk.img_mod_embed
        return x

    def call_block(self, blk, x):
        h, w = self._grid
        return blk(x, mask=None, T=1, H_patches=h, W_patches=w, mode="img")[0]

    def final(self, x):
        return self.trunk.norms_block[-1](x)


def load_backbone(weights=RELEASED_ENCODER):
    """Frozen V-JEPA 2.1 ViT-L from an encoder state dict (relative paths are under $AI4EYES_CHECKPOINT_DIR):
    the released one, or the EMA target encoder exported after SSL (scripts/export_encoder.sh)."""
    path = paths.CHECKPOINT_DIR / weights
    enc = build_encoder(384)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = enc.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    return FrozenVJEPA(enc)
