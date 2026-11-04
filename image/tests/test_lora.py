import torch
import torch.nn as nn

from ai4eyes_image.models.lora import LoRAQV, LoRAViT, trainable_state


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = _Attn(d)

    def forward(self, x):
        q, k, v = self.attn.qkv(x).chunk(3, -1)
        return x + q * v


class _Frozen(nn.Module):
    """Stand-in for FrozenVJEPA: tiny trunk with the same embed / call_block / final interface."""
    patch_size = 16

    def __init__(self, d=8, n=3):
        super().__init__()
        self.trunk = nn.Module()
        self.trunk.blocks = nn.ModuleList([_Block(d) for _ in range(n)])
        self.trunk.norm = nn.LayerNorm(d)
        self.embed_dim = d

    def embed(self, x):
        return x

    def call_block(self, blk, x):
        return blk(x)

    def final(self, x):
        return self.trunk.norm(x)


def test_lora_starts_at_the_frozen_layer():
    base = nn.Linear(8, 24)
    x = torch.randn(2, 5, 8)
    assert torch.allclose(LoRAQV(base, rank=2)(x), base(x))


def test_only_lora_and_head_train():
    m = LoRAViT(None, n_classes=3, rank=2, frozen=_Frozen(), lora_blocks=2)
    assert m.backbone_frozen()
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names and all(".A_" in n or ".B_" in n or n.startswith("head.") for n in names)
    assert not any("blocks.0." in n for n in names)                    # lora_blocks=2 → only the last 2 blocks
    assert set(trainable_state(m)) == names | {"feat_mean", "feat_scale"}


def test_masked_pooling_ignores_dropped_tokens():
    m = LoRAViT(None, n_classes=3, frozen=_Frozen()).eval()
    x = torch.randn(1, 4, 8)
    keep = torch.tensor([[True, True, False, False]])
    x2 = x.clone()
    x2[:, 2:] = 100                                                    # tokens outside the mask may be anything
    # the tiny trunk mixes nothing across tokens, so pooled features depend only on the kept tokens
    assert torch.allclose(m.pooled(x, keep), m.pooled(x2, keep), atol=1e-5)
    assert m(x, keep).shape == (1, 3)
