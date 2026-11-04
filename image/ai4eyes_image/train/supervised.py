"""Supervised training of every image sign on the shared train / val / test split.

    python -m ai4eyes_image.train.supervised run configs/supervised.yaml [--signs ...] [--max-epochs N] [--seeds 0,1]

Per sign: 5 seeds are trained on the train patients for train.max_epochs. After every epoch the val and test
images are predicted and persisted, and the trainable weights (LoRA + head) are saved to $AI4EYES_WORK_DIR. The
epoch is chosen on VAL only (mean over seeds of val macro-F1, or val MAE for tear meniscus, 3-epoch smoothed);
the 5 checkpoints of that epoch are kept in $AI4EYES_CHECKPOINT_DIR/signs/<sign>/. Test predictions at that epoch
are read once, later, by ai4eyes_image.eval.report — nothing here looks at test metrics.

Outputs: $AI4EYES_OUTPUT_DIR/predictions/<sign>.parquet (val + test rows, every seed and epoch, `selected_epoch`),
$AI4EYES_CHECKPOINT_DIR/signs/<sign>/seed<k>.pt + meta.json. A sign whose outputs exist is skipped (resumable).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import logging
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from ai4eyes_image import paths
from ai4eyes_image.config import load_yaml, sign_config
from ai4eyes_image.data.split import load_split
from ai4eyes_image.eval.metrics import cls_metrics
from ai4eyes_image.models.lora import trainable_state
from ai4eyes_image.preprocess.banks import make_bank
from ai4eyes_image.signs import SIGNS, labels
from ai4eyes_image.train.trainer import predict, setup, train_epoch

log = logging.getLogger("supervised")


def pred_path(sign):
    return paths.OUTPUT_DIR / "predictions" / f"{sign}.parquet"


def ckpt_dir(sign):
    return paths.CHECKPOINT_DIR / "signs" / sign


def data(sign, df):
    """Label rows of `sign` with their patient's split."""
    d = labels(sign, df)
    d["split"] = d.patient_pid.map(load_split())
    assert d.split.notna().all(), f"{sign}: patients missing from {paths.SPLIT_CSV}"
    assert d.sample_id.is_unique
    return d


def select_epoch(pr, kind, smooth):
    """Epoch chosen on val only → (best epoch, per-epoch val curve)."""
    v = pr[pr.split == "val"]
    if kind == "regression":
        per = v.groupby(["epoch", "seed"]).apply(lambda g: np.abs(g.pred_mm - g.y).mean()).groupby("epoch").mean()
        return int(per.rolling(smooth, center=True, min_periods=1).mean().idxmin()), per
    pc = [x for x in pr.columns if x.startswith("p_")]
    per = v.groupby(["epoch", "seed"]).apply(lambda g: cls_metrics(g.y.to_numpy(), g[pc].to_numpy())["f1"]) \
           .groupby("epoch").mean()
    return int(per.rolling(smooth, center=True, min_periods=1).mean().idxmax()), per


def run_sign(sign, df, cfg, device, seeds, max_epochs=None):
    kind = SIGNS[sign].kind
    if pred_path(sign).exists() and all((ckpt_dir(sign) / f"seed{s}.pt").exists() for s in seeds):
        log.info("%s: done, skipped", sign)
        return
    c = sign_config(cfg, sign)
    if max_epochs:
        c["train"]["max_epochs"] = max_epochs
    d = data(sign, df)
    tr, ev = d[d.split == "train"].copy(), d[d.split != "train"]
    mu = sd = None
    if kind == "regression":
        mu, sd = float(tr.y.mean()), float(tr.y.std())
        tr["y"] = (tr.y - mu) / sd
    bank = make_bank(sign, d)
    n_ep = c["train"]["max_epochs"]
    work = paths.WORK_DIR / "supervised" / sign
    rows = []
    for seed in seeds:
        t0 = time.time()
        st = setup(c, bank, tr, 10_000 + seed, device)
        for ep in range(n_ep):
            train_epoch(st, c, bank, list(tr.sample_id), tr.y.to_numpy(), ep, device)
            z = predict(st["model"], bank, list(ev.sample_id), device, c["train"]["batch_size"], st["amp"])
            r = ev[["sample_id", "patient_pid", "exam_id", "eye", "split", "y"]].copy()
            r["seed"], r["epoch"] = seed, ep
            if kind == "regression":
                r["pred_mm"] = z.squeeze(1).float().numpy() * sd + mu
            else:
                P = torch.softmax(z.float(), 1).numpy()
                for k in range(P.shape[1]):
                    r[f"p_{k}"] = P[:, k]
            rows.append(r)
            p = work / f"seed{seed}_ep{ep}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trainable_state(st["model"]), p)
        log.info("%s seed %d: %d epochs on %d train rows, %.1f min", sign, seed, n_ep, len(tr), (time.time() - t0) / 60)
        del st
        torch.cuda.empty_cache()
    pr = pd.concat(rows, ignore_index=True)
    best, per = select_epoch(pr, kind, cfg.get("smooth_epochs", 3))
    out = ckpt_dir(sign)
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        shutil.copy2(work / f"seed{seed}_ep{best}.pt", out / f"seed{seed}.pt")
    meta = {"sign": sign, "kind": kind, "epochs": best + 1,
            "selected_on": f"val (mean over seeds, {cfg.get('smooth_epochs', 3)}-epoch smoothed)",
            "val_curve": {int(k): float(x) for k, x in per.items()}, "seeds": list(seeds), "train_seed": "10000 + seed",
            "train_rows": len(tr), "train_patients": int(tr.patient_pid.nunique()), "config": c}
    if kind == "regression":
        meta.update(train_mean_mm=mu, train_sd_mm=sd)
    else:
        meta["classes"] = list(SIGNS[sign].classes)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    pr["selected_epoch"] = best
    pred_path(sign).parent.mkdir(parents=True, exist_ok=True)
    pr.to_parquet(pred_path(sign), index=False)
    shutil.rmtree(work, ignore_errors=True)
    log.info("%s: selected epoch %d on val → %s", sign, best + 1, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("config")
    ap.add_argument("--signs", nargs="*")
    ap.add_argument("--max-epochs", type=int, help="smoke test: fewer epochs")
    ap.add_argument("--seeds", default=None, help="comma-separated, default: the config's seeds")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = load_yaml(a.config)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["seeds"]
    df = pd.read_csv(paths.LABELS_CSV, low_memory=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for s in a.signs or list(SIGNS):
        run_sign(s, df, cfg, device, seeds, a.max_epochs)


if __name__ == "__main__":
    main()
