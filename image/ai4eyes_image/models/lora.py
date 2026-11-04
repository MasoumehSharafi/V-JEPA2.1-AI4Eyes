"""LoRA + linear head on a frozen V-JEPA 2.1 encoder, trained end-to-end (random head, LoRA from step 0).

LoRA adapts attention q and v only: every block has one fused `qkv` Linear, so low-rank deltas are added to the
q and v slices of its output (k is untouched). Every original backbone weight keeps requires_grad=False; only the
LoRA matrices and the head train. Saved checkpoints hold just those (+ the two feature buffers).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai4eyes_image.models.vjepa import load_backbone


class LoRAQV(nn.Module):
    """Wraps a fused qkv Linear(C, 3C): out = W x + b + s·[B_q A_q x, 0, B_v A_v x]."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        C = base.in_features
        assert base.out_features == 3 * C
        self.C, self.scale = C, alpha / rank
        self.drop = nn.Dropout(dropout)
        self.A_q, self.B_q = nn.Linear(C, rank, bias=False), nn.Linear(rank, C, bias=False)
        self.A_v, self.B_v = nn.Linear(C, rank, bias=False), nn.Linear(rank, C, bias=False)
        for A, B in ((self.A_q, self.B_q), (self.A_v, self.B_v)):
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
            nn.init.zeros_(B.weight)  # delta = 0 at init → starts exactly at the frozen backbone

    def forward(self, x):
        out = self.base(x)
        h = self.drop(x)
        dq = self.B_q(self.A_q(h)) * self.scale
        dv = self.B_v(self.A_v(h)) * self.scale
        q, k, v = out.split(self.C, dim=-1)
        return torch.cat([q + dq, k, v + dv], dim=-1)


class LoRAViT(nn.Module):
    def __init__(self, backbone_weights, n_classes, rank=8, alpha=16, lora_dropout=0.0, lora_blocks=None,
                 pool="masked", head_dropout=0.0, grad_checkpoint=False, frozen=None):
        """lora_blocks: adapt only the last k transformer blocks (None = all). n_classes=1 → regression.
        frozen: an already-loaded FrozenVJEPA (for tests); default loads `backbone_weights`."""
        super().__init__()
        frozen = frozen if frozen is not None else load_backbone(backbone_weights)
        # Plain attribute, NOT a submodule: the wrapper's train() pins eval mode and would re-register
        # the trunk. Only its embed()/call_block()/final() helpers are used; the trunk is registered below.
        object.__setattr__(self, "frozen", frozen)
        self.trunk, self.embed_dim, self.patch_size = frozen.trunk, frozen.embed_dim, frozen.patch_size
        self.pool, self.grad_checkpoint = pool, grad_checkpoint
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        n_blocks = len(self.trunk.blocks)
        first_adapted = 0 if lora_blocks is None else n_blocks - int(lora_blocks)
        assert 0 <= first_adapted < n_blocks, f"lora_blocks={lora_blocks} with {n_blocks} blocks"
        for i, blk in enumerate(self.trunk.blocks):
            if i >= first_adapted:
                blk.attn.qkv = LoRAQV(blk.attn.qkv, rank=rank, alpha=alpha, dropout=lora_dropout)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.embed_dim, n_classes)
        # fixed feature normalisation, off (kept so checkpoints load with the same keys)
        self.feat_norm = False
        self.register_buffer("feat_mean", torch.zeros(self.embed_dim))
        self.register_buffer("feat_scale", torch.ones(self.embed_dim))

    def adapter_parameters(self):
        return [p for n, p in self.trunk.named_parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.head.parameters())

    def param_counts(self):
        return {"adapter": sum(p.numel() for p in self.adapter_parameters()),
                "head": sum(p.numel() for p in self.head_parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": sum(p.numel() for p in self.parameters())}

    def backbone_frozen(self):
        """True iff every original backbone weight is frozen (only LoRA A/B may train)."""
        return all(not p.requires_grad or any(f".{m}." in n for m in ("A_q", "B_q", "A_v", "B_v"))
                   for n, p in self.trunk.named_parameters())

    def tokens(self, x):
        x = self.frozen.embed(x)
        for blk in self.trunk.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(self.frozen.call_block, blk, x, use_reentrant=False)
            else:
                x = self.frozen.call_block(blk, x)
        return self.frozen.final(x)            # [B, T, D] patch tokens

    def pooled(self, x, keep=None):
        tok = self.tokens(x)
        if self.pool == "masked":
            assert keep is not None
            w = keep.to(tok.dtype).unsqueeze(-1)
            return (tok * w).sum(1) / w.sum(1).clamp_min(1)
        if self.pool == "all":
            return tok.mean(1)
        raise ValueError(self.pool)

    def forward(self, x, keep=None):
        feat = self.pooled(x, keep)
        if self.feat_norm:
            feat = (F.normalize(feat.float(), dim=1) - self.feat_mean) / self.feat_scale
        return self.head(self.head_drop(feat))


def trainable_state(model):
    """Only what training changes: LoRA + head weights and the buffers (the frozen backbone is not copied)."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
