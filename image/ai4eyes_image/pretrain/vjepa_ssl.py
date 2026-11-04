"""Continued SSL pretraining of the V-JEPA 2.1 ViT-L/16 encoder on the pooled unlabelled sign images.

    python -m ai4eyes_image.pretrain.vjepa_ssl configs/pretrain_ssl.yaml --out-dir <dir> [--mirror-dir <dir>] [--resume]
    python -m ai4eyes_image.pretrain.vjepa_ssl export <last.pt> <encoder.pt>     # EMA target encoder → backbone

Data: $AI4EYES_SSL_MANIFEST (ai4eyes_image.pretrain.manifest); every exam of the labels CSV is excluded (checked
here again). All rows are used; there is no held-out selection, only the last checkpoint is kept.

Objective = V-JEPA 2.1 on images, self-distillation form: the context encoder sees the visible tokens of a
multi-block mask (10 blocks × 15 % of the image, aspect 0.75–1.5); the predictor predicts the target encoder's
(EMA of the context encoder) layer-normalised final tokens at the masked positions (L1), plus the V-JEPA 2.1 dense
context loss on the visible positions (weight 0.5). The released ViT-L was distilled from ViT-G, so its predictor
outputs 1664-d features: the predictor body is kept, its two output projections are replaced by fresh 1024-d ones,
trained alone for `warmup_pred_epochs` before the encoder unfreezes. Resolution 384 (native).

Checkpoint: ONE file, <out-dir>/<name>_last.pt (target encoder, context encoder, predictor, optimiser, scaler,
epoch), overwritten every `save_every` epochs and at the end, so --resume continues from it; with --mirror-dir it
is also copied there every `mirror_every` epochs (e.g. from a local NVMe disk that is wiped on shutdown to
persistent storage). Per-epoch loss log: <name>_log.csv.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import logging
import math
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ai4eyes_image import paths
from ai4eyes_image.models.vjepa import PATCH, RELEASED_ENCODER, RELEASED_PREDICTOR, build_encoder, vjepa_modules
from ai4eyes_image.pretrain.dataset import SSLImages

log = logging.getLogger("vjepa_ssl")


def _atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# ---- model ---------------------------------------------------------------------------------------
def build_models(res):
    """(context encoder, predictor) initialised from the released V-JEPA 2.1 ViT-L/16 384 weights."""
    _, vp = vjepa_modules()
    enc = build_encoder(res)
    sd = torch.load(paths.CHECKPOINT_DIR / RELEASED_ENCODER, map_location="cpu")
    missing, unexpected = enc.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    pred = vp.vit_predictor(img_size=(res, res), patch_size=PATCH, use_mask_tokens=True, embed_dim=1024,
                            predictor_embed_dim=384, teacher_embed_dim=1664, num_frames=64, tubelet_size=2,
                            depth=12, num_heads=12, num_mask_tokens=8, use_rope=True, uniform_power=False,
                            use_sdpa=True, use_silu=False, wide_silu=True, n_output_distillation=1,
                            return_all_tokens=True, img_temporal_dim_size=1)
    sd = torch.load(paths.CHECKPOINT_DIR / RELEASED_PREDICTOR, map_location="cpu")
    missing, unexpected = pred.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    # ViT-G (1664-d) output heads → fresh ViT-L (1024-d) heads, V-JEPA's own init
    for name in ("predictor_proj", "predictor_proj_context"):
        lin = nn.Linear(384, 1024)
        nn.init.trunc_normal_(lin.weight, std=0.02)
        nn.init.zeros_(lin.bias)
        setattr(pred, name, lin)
    return enc, pred


def encode(enc, x, keep=None):
    """Final normalised tokens of the image path. keep: [B, K] indices of visible tokens (None = all)."""
    g = x.shape[-1] // PATCH
    t = enc.patch_embed_img(x.unsqueeze(2))
    if enc.modality_embedding:
        t = t + enc.img_mod_embed
    m = None
    if keep is not None:
        t = torch.gather(t, 1, keep.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
        m = keep
    for blk in enc.blocks:
        t = blk(t, mask=m, T=1, H_patches=g, W_patches=g, mode="img")[0]
    return enc.norms_block[-1](t)


# ---- masks ---------------------------------------------------------------------------------------
def multiblock_masks(batch, g, rng, n_blocks=10, scale=0.15, aspect=(0.75, 1.5)):
    """V-JEPA multi-block mask for images: union of n_blocks rectangles (each `scale` of the image) = target;
    the rest = context. Counts are equalised over the batch (random subset down to the minimum), as V-JEPA's
    collator does, so both are dense [B, K] index tensors."""
    ctx, tgt = [], []
    for _ in range(batch):
        m = np.zeros((g, g), bool)
        for _ in range(n_blocks):
            ar = math.exp(rng.uniform(math.log(aspect[0]), math.log(aspect[1])))
            h = int(round(math.sqrt(scale * g * g * ar)))
            w = int(round(math.sqrt(scale * g * g / ar)))
            h, w = min(max(h, 1), g), min(max(w, 1), g)
            y, x = rng.integers(0, g - h + 1), rng.integers(0, g - w + 1)
            m[y:y + h, x:x + w] = True
        idx = np.arange(g * g)
        tgt.append(idx[m.ravel()])
        ctx.append(idx[~m.ravel()])
    kc, kt = min(map(len, ctx)), min(map(len, tgt))
    assert kc > 0 and kt > 0
    ctx = [np.sort(rng.choice(c, kc, replace=False)) for c in ctx]
    tgt = [np.sort(rng.choice(t, kt, replace=False)) for t in tgt]
    return torch.as_tensor(np.stack(ctx)), torch.as_tensor(np.stack(tgt))


# ---- training ------------------------------------------------------------------------------------
def lr_at(step, total, warm, peak, final):
    if step < warm:
        return peak * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * p))


def param_groups(module, wd):
    decay, no_decay = [], []
    for _, p in module.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def jepa_loss(enc, target, pred, x, ctx, tgt, ctx_weight, amp):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        with torch.no_grad():
            h = target_tokens(target, x)
        z = encode(enc, x, ctx)
        z_pred, z_ctx = pred(z, [ctx], [tgt], mod="image", mask_index=0)
    h_t = torch.gather(h, 1, tgt.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
    h_c = torch.gather(h, 1, ctx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
    l_pred = (z_pred.float() - h_t).abs().mean()
    l_ctx = (z_ctx.float() - h_c).abs().mean()
    return l_pred + ctx_weight * l_ctx, l_pred, l_ctx


def target_tokens(target, x):
    h = encode(target, x)
    return F.layer_norm(h.float(), (h.shape[-1],))


def mirror(src, dst_dir):
    """Copy a file into dst_dir atomically (tmp + rename), so a stop mid-copy never leaves a broken file."""
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    return dst


def main(config_path, out_dir=".", resume=False, max_steps=None, mirror_dir=None):
    cfg = yaml.safe_load(open(config_path))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    last_p = os.path.join(out_dir, f"{cfg['name']}_last.pt")
    log_p = os.path.join(out_dir, f"{cfg['name']}_log.csv")
    torch.manual_seed(cfg["seed"])
    device, amp = "cuda", bool(cfg.get("amp", True))
    res, o = cfg["res"], cfg["optim"]

    manifest = pd.read_csv(paths.SSL_MANIFEST)
    labelled = set(pd.read_csv(paths.LABELS_CSV, usecols=["exam_id"]).exam_id)
    assert not set(manifest.exam_id) & labelled, "a labelled exam is in the pretraining manifest"
    ds = SSLImages(manifest, paths.S3_BUCKET, res, cfg["augment"], train=True, seed=cfg["seed"])
    nw = cfg["data"].get("num_workers", 8)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=nw, pin_memory=True,
                    persistent_workers=nw > 0, prefetch_factor=4 if nw else None,
                    generator=torch.Generator().manual_seed(cfg["seed"]))
    spe = len(dl)
    total = cfg["epochs"] * spe
    log.info("pretraining images: %d (%d exams), %d steps/epoch, %d epochs", len(manifest),
             manifest.exam_id.nunique(), spe, cfg["epochs"])

    enc, pred = build_models(res)
    target = copy.deepcopy(enc)
    for p in target.parameters():
        p.requires_grad_(False)
    enc, pred, target = enc.to(device), pred.to(device), target.to(device)
    opt = torch.optim.AdamW(
        [{**g, "base": "enc"} for g in param_groups(enc, o["weight_decay"])]
        + [{**g, "base": "pred"} for g in param_groups(pred, o["weight_decay"])],
        lr=0.0, betas=tuple(o["betas"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, hist = 0, []
    if resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck["encoder"])
        target.load_state_dict(ck["target_encoder"])
        pred.load_state_dict(ck["predictor"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"] + 1
        hist = pd.read_csv(log_p).to_dict("records") if os.path.exists(log_p) else []
        log.info("resumed from %s at epoch %d", last_p, start)

    def save(ep):
        _atomic_save({"target_encoder": target.state_dict(), "encoder": enc.state_dict(),
                      "predictor": pred.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                      "epoch": ep, "config": cfg, "n_images": len(manifest)}, last_p)

    rng = np.random.default_rng([cfg["seed"], start])
    g = res // PATCH
    warm_pred = cfg.get("warmup_pred_epochs", 0)
    steps = 0
    for ep in range(start, cfg["epochs"]):
        t0 = time.time()
        ds.set_epoch(ep)
        train_enc = ep >= warm_pred
        for p in enc.parameters():
            p.requires_grad_(train_enc)
        enc.train(train_enc)
        pred.train()
        tot = {"loss": 0.0, "pred": 0.0, "ctx": 0.0}
        n = 0
        for i, x in enumerate(dl):
            step = ep * spe + i
            lr_e = lr_at(step, total, o["warmup_epochs"] * spe, o["lr"], o["final_lr"]) if train_enc else 0.0
            lr_p = lr_at(step, total, o["warmup_epochs"] * spe, o["lr_predictor"], o["final_lr"])
            for gp in opt.param_groups:
                gp["lr"] = lr_e if gp["base"] == "enc" else lr_p
            x = x.to(device, non_blocking=True)
            ctx, tgt = (t.to(device) for t in multiblock_masks(len(x), g, rng, **cfg["mask"]))
            loss, lp, lc = jepa_loss(enc, target, pred, x, ctx, tgt, cfg["ctx_loss_weight"], amp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep} step {i}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for gp in opt.param_groups for p in gp["params"]], o["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if train_enc:   # EMA target update
                m = cfg["ema"]
                with torch.no_grad():
                    for pt, pc in zip(target.parameters(), enc.parameters()):
                        pt.mul_(m).add_(pc.detach(), alpha=1 - m)
            for k, v in (("loss", loss), ("pred", lp), ("ctx", lc)):
                tot[k] += v.item() * len(x)
            n += len(x)
            steps += 1
            if max_steps and steps >= max_steps:
                torch.cuda.synchronize()
                sec = time.time() - t0
                log.info("benchmark: %d steps of %d images in %.1f s → %.2f s/step, %.0f s/epoch (%d steps), "
                         "peak GPU memory %.1f GB", steps, cfg["batch_size"], sec, sec / steps, sec / steps * spe,
                         spe, torch.cuda.max_memory_allocated() / 2 ** 30)
                return
        sec = time.time() - t0
        hist.append({"epoch": ep, "loss": tot["loss"] / n, "loss_pred": tot["pred"] / n, "loss_ctx": tot["ctx"] / n,
                     "lr_encoder": lr_e, "lr_predictor": lr_p, "encoder_trained": train_enc, "sec": sec})
        pd.DataFrame(hist).to_csv(log_p, index=False)
        if (ep + 1) % cfg.get("save_every", 1) == 0 or ep == cfg["epochs"] - 1:
            save(ep)
        if mirror_dir and ((ep + 1) % cfg.get("mirror_every", 25) == 0 or ep == cfg["epochs"] - 1):
            log.info("epoch %d: copied the last checkpoint → %s", ep, mirror(last_p, mirror_dir))
            mirror(log_p, mirror_dir)
        log.info("epoch %d/%d  loss %.4f (pred %.4f, ctx %.4f)  lr enc %.2e pred %.2e%s  %.0f s", ep,
                 cfg["epochs"] - 1, tot["loss"] / n, tot["pred"] / n, tot["ctx"] / n, lr_e, lr_p,
                 "" if train_enc else "  [predictor only]", sec)
    log.info("done → %s (target encoder = the backbone)", last_p)


def export(last_p, enc_p):
    """EMA target encoder of a FINISHED run → state dict loadable by ai4eyes_image.models.vjepa.load_backbone."""
    ck = torch.load(last_p, map_location="cpu", weights_only=False)
    assert ck["epoch"] == ck["config"]["epochs"] - 1, f"pretraining not finished (epoch {ck['epoch']})"
    os.makedirs(os.path.dirname(os.path.abspath(enc_p)), exist_ok=True)
    torch.save(ck["target_encoder"], enc_p)
    enc = build_encoder(384)
    missing, unexpected = enc.load_state_dict(torch.load(enc_p, map_location="cpu", weights_only=True), strict=True)
    assert not missing and not unexpected
    log.info("exported + verified %s", enc_p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3])
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, help="benchmark: stop after N steps and report the speed")
    ap.add_argument("--mirror-dir", help="also copy the last checkpoint here every `mirror_every` epochs")
    a = ap.parse_args()
    if not paths.S3_BUCKET:
        log.warning("AI4EYES_S3_BUCKET is not set: only manifest rows with local_path can be read")
    main(a.config, a.out_dir, a.resume, a.max_steps, a.mirror_dir)
