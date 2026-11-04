"""Seeded training of LoRAViT: optimiser + cosine schedule (setup), one epoch (train_epoch), inference (predict).

Classification: cross-entropy with sqrt-inverse-frequency class weights (capped at 10×) over the classes present
in the training set. Regression (`train.loss: mse`, 1 output): MSE on the standardised target.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must precede CUDA init

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from ai4eyes_image.models.lora import LoRAViT

WEIGHT_CAP = 10.0


def set_seed(s):
    s = int(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights(y, scheme):
    """sqrt-inverse-frequency capped at WEIGHT_CAP, over classes present in this training set; or None."""
    if scheme in (None, "none"):
        return None
    if scheme != "sqrt_inv":
        raise ValueError(scheme)
    classes, counts = np.unique(y, return_counts=True)
    n_total, n_classes = len(y), len(classes)
    return {int(c): min(WEIGHT_CAP, float(np.sqrt(n_total / (n_classes * n)))) for c, n in zip(classes, counts)}


def lr_lambda(warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def build_model(cfg, device):
    m = cfg["model"]
    return LoRAViT(m["backbone_weights"], n_classes=m["n_classes"], rank=m.get("rank", 8), alpha=m.get("alpha", 16),
                   lora_dropout=m.get("lora_dropout", 0.0), lora_blocks=m.get("lora_blocks"),
                   pool=m.get("pool", "masked"), head_dropout=m.get("head_dropout", 0.0),
                   grad_checkpoint=m.get("grad_checkpoint", False)).to(device)


@torch.no_grad()
def predict(model, bank, ids, device, bs, amp):
    """Raw outputs (logits, or the standardised value for regression) [N, n_out], no augmentation."""
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        x, k = bank.eval_batch(ids[i:i + bs])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out.append(model(x.to(device), k.to(device)).float().cpu())
    return torch.cat(out)


def setup(cfg, bank, df_tr, seed, device):
    """Seeded model + AdamW (adapter / head groups) + linear-warmup cosine schedule over train.max_epochs."""
    t = cfg["train"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(cfg, device)
    assert model.patch_size == bank.patch, "image bank token grid does not match the backbone patch size"
    assert model.backbone_frozen()
    amp = bool(t.get("amp", True)) and device == "cuda"

    cw = torch.ones(model.head.out_features)
    w = None if t.get("loss") == "mse" else class_weights(df_tr.y.to_numpy(), t.get("class_weight", "sqrt_inv"))
    for c, v in (w or {}).items():
        cw[c] = v
    cw = cw.to(device)

    decay = lambda ps: [p for p in ps if p.ndim > 1]
    no_decay = lambda ps: [p for p in ps if p.ndim <= 1]
    A, H = model.adapter_parameters(), model.head_parameters()
    wd_a = t.get("weight_decay_adapter", t["weight_decay"])
    opt = torch.optim.AdamW([
        {"params": decay(A), "lr": t["lr_adapter"], "weight_decay": wd_a},
        {"params": no_decay(A), "lr": t["lr_adapter"], "weight_decay": 0.0},
        {"params": decay(H), "lr": t["lr_head"], "weight_decay": t["weight_decay"]},
        {"params": no_decay(H), "lr": t["lr_head"], "weight_decay": 0.0},
    ])
    steps_per_epoch = math.ceil(len(df_tr) / t["batch_size"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(t["warmup_epochs"] * steps_per_epoch, t["max_epochs"] * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    return {"model": model, "opt": opt, "sched": sched, "scaler": scaler, "cw": cw, "rng": rng, "amp": amp}


def train_epoch(st, cfg, bank, tr_ids, y_tr, ep, device):
    t, model, opt, amp = cfg["train"], st["model"], st["opt"], st["amp"]
    freeze_head = ep < t.get("head_freeze_epochs", 0)
    for p in model.head_parameters():
        p.requires_grad_(not freeze_head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    bs, rng = t["batch_size"], st["rng"]
    model.train()
    order = rng.permutation(len(tr_ids))
    tot, n = 0.0, 0
    for b in range(0, len(order), bs):
        ids = [tr_ids[i] for i in order[b:b + bs]]
        yb = torch.as_tensor(y_tr[order[b:b + bs]], device=device)
        x, k = bank.train_batch(ids, rng, cfg.get("augment"))
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x.to(device), k.to(device))
        if t.get("loss") == "mse":
            loss = F.mse_loss(logits.float().squeeze(1), yb.float())
        else:
            loss = F.cross_entropy(logits.float(), yb, weight=st["cw"], label_smoothing=t.get("label_smoothing", 0.0))
        opt.zero_grad(set_to_none=True)
        st["scaler"].scale(loss).backward()
        if t.get("grad_clip"):
            st["scaler"].unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, t["grad_clip"])
        st["scaler"].step(opt)
        st["scaler"].update()
        st["sched"].step()
        tot, n = tot + loss.item() * len(ids), n + len(ids)
    for p in model.head_parameters():
        p.requires_grad_(True)
    return tot / n
