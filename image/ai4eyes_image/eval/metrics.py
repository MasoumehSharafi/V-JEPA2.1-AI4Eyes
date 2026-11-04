"""Test metrics, always computed from persisted prediction files (never inside training).

Classification: accuracy + macro precision / recall / F1 over the classes present in y. Regression (tear meniscus,
mm): MAE, RMSE, share within ±0.05 mm, Pearson r, agreement at the 0.2 mm dry-eye cut-off. Confidence intervals:
patient-level bootstrap (both eyes and every label row of a resampled patient stay together).
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

N_BOOT = 2000


def cls_metrics(y, P):
    yp = P.argmax(1)
    present = sorted(set(y))
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=present, average="macro", zero_division=0)
    return {"accuracy": float((y == yp).mean()), "precision": p, "recall": r, "f1": f}


def reg_metrics(y, yp):
    e = np.abs(yp - y)
    return {"mae_mm": float(e.mean()), "rmse_mm": float(np.sqrt(((yp - y) ** 2).mean())),
            "within_0_05mm": float((e <= 0.05).mean()), "pearson_r": float(np.corrcoef(y, yp)[0, 1]),
            "agree_0_2mm": float(((y <= 0.2) == (yp <= 0.2)).mean()), "true_le_0_2": int((y <= 0.2).sum()),
            "pred_le_0_2": int((yp <= 0.2).sum())}


def tear_extra(y, p):
    lo_t, lo_p = y <= 0.2, p <= 0.2
    return {"sensitivity_le_0_2mm": float((lo_t & lo_p).sum() / lo_t.sum()) if lo_t.sum() else np.nan,
            "specificity_le_0_2mm": float((~lo_t & ~lo_p).sum() / (~lo_t).sum()) if (~lo_t).sum() else np.nan}


def boot_ci(e, fn, n=N_BOOT, seed=0):
    """95 % percentile CI of fn(rows) under a patient-level bootstrap. e: one row per label row, with patient_pid."""
    rng = np.random.default_rng(seed)
    groups = list(e.groupby("patient_pid").indices.values())
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(fn(e.iloc[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return lo, hi
