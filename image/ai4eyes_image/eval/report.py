"""Test results of every sign, from the persisted predictions (test rows at the val-chosen epoch).

    python -m ai4eyes_image.eval.report

Writes to $AI4EYES_OUTPUT_DIR:
  results.md                         one table for the classification signs, one for tear meniscus
  test_metrics_classification.csv / test_metrics_tear_meniscus.csv
  <sign>_test_pred_vs_gt.csv         one row per test label row (image_id = <exam_id>-<eye>), prediction + truth
  metrics/<sign>/summary.json        5-seed-ensemble metrics, patient-level bootstrap 95 % CIs, per-seed mean ± sd
  metrics/<sign>/per_class.csv, confusion.csv (classification), per_seed.csv
  metrics/all_tests.csv              one row per sign
Ensemble = mean of the 5 seeds' probabilities (classification) or mm (regression).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ai4eyes_image import paths
from ai4eyes_image.eval.metrics import boot_ci, cls_metrics, reg_metrics, tear_extra
from ai4eyes_image.signs import SIGNS, labels
from ai4eyes_image.train.supervised import pred_path


def test_rows(sign):
    pr = pd.read_parquet(pred_path(sign))
    best = int(pr.selected_epoch.iloc[0])
    return pr[(pr.split == "test") & (pr.epoch == best)], best


def ensemble(t, pc=None):
    agg = {"patient_pid": ("patient_pid", "first"), "y": ("y", "first")}
    agg.update({c: (c, "mean") for c in pc} if pc else {"pred_mm": ("pred_mm", "mean")})
    return t.groupby("sample_id").agg(**agg)


def store(sign):
    """metrics/<sign>/ files → the summary dict."""
    t, best = test_rows(sign)
    d = paths.OUTPUT_DIR / "metrics" / sign
    d.mkdir(parents=True, exist_ok=True)
    s = {"test": sign, "epochs_chosen_on_val": best + 1, "seeds": int(t.seed.nunique()),
         "test_patients": int(t.patient_pid.nunique()), "test_eyes": int(t[["exam_id", "eye"]].drop_duplicates().shape[0]),
         "test_label_rows": int(t.sample_id.nunique())}
    if SIGNS[sign].kind == "regression":
        e = ensemble(t)
        y, p = e.y.to_numpy(), e.pred_mm.to_numpy()
        s.update({k: float(v) for k, v in {**reg_metrics(y, p), **tear_extra(y, p)}.items()})
        for k in ("mae_mm", "within_0_05mm", "pearson_r"):
            lo, hi = boot_ci(e, lambda x, k=k: reg_metrics(x.y.to_numpy(), x.pred_mm.to_numpy())[k])
            s[f"{k}_ci95"] = [float(lo), float(hi)]
        seeds = pd.DataFrame([{"seed": k, **reg_metrics(g.y.to_numpy(), g.pred_mm.to_numpy()),
                               **tear_extra(g.y.to_numpy(), g.pred_mm.to_numpy())} for k, g in t.groupby("seed")])
    else:
        names = list(SIGNS[sign].classes)
        pc = [c for c in t.columns if c.startswith("p_")]
        e = ensemble(t, pc)
        y, P = e.y.to_numpy(), e[pc].to_numpy()
        yp = P.argmax(1)
        s.update({k: float(v) for k, v in cls_metrics(y, P).items()})
        s["classes"] = names
        for k in ("accuracy", "precision", "recall", "f1"):
            lo, hi = boot_ci(e, lambda x, k=k: cls_metrics(x.y.to_numpy(), x[pc].to_numpy())[k])
            s[f"{k}_ci95"] = [float(lo), float(hi)]
        lab = list(range(len(names)))
        pr_, rc_, f1_, sup = precision_recall_fscore_support(y, yp, labels=lab, zero_division=0)
        pd.DataFrame({"class": names, "precision": pr_, "recall": rc_, "f1": f1_, "support": sup,
                      "predicted": np.bincount(yp, minlength=len(names))}).round(4).to_csv(d / "per_class.csv", index=False)
        pd.DataFrame(confusion_matrix(y, yp, labels=lab), index=[f"true {n}" for n in names],
                     columns=[f"pred {n}" for n in names]).to_csv(d / "confusion.csv")
        seeds = pd.DataFrame([{"seed": k, **cls_metrics(g.y.to_numpy(), g[pc].to_numpy())} for k, g in t.groupby("seed")])
    seeds.round(4).to_csv(d / "per_seed.csv", index=False)
    for c in seeds.columns.drop("seed"):
        if np.issubdtype(seeds[c].dtype, np.number):
            s[f"{c}_seed_mean"], s[f"{c}_seed_sd"] = float(seeds[c].mean()), float(seeds[c].std(ddof=1))
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def export(sign, df, t):
    """Test rows at the chosen epoch → <sign>_test_pred_vs_gt.csv."""
    d = labels(sign, df).set_index("sample_id")
    g = t.groupby("sample_id")
    e = g[["patient_pid", "exam_id", "eye", "y"]].first()
    out = pd.DataFrame({"image_id": e.exam_id + "-" + e.eye, "patient_pid": e.patient_pid, "exam_id": e.exam_id,
                        "eye": e.eye, "labeller_pid": d.labeller_pid.reindex(e.index),
                        "gt_gradable": d.eye_status.reindex(e.index).ne("REJECTED"),
                        "gt_label_raw": d.label_raw.reindex(e.index)})
    if SIGNS[sign].kind == "regression":
        pm = g.pred_mm.mean()
        out = out.assign(gt_mm=e.y, pred_mm=pm, error_mm=pm - e.y, abs_error_mm=(pm - e.y).abs(),
                         within_0_05mm=(pm - e.y).abs() <= 0.05, gt_le_0_2mm=e.y <= 0.2, pred_le_0_2mm=pm <= 0.2)
    else:
        pc = [c for c in t.columns if c.startswith("p_")]
        names = dict(enumerate(SIGNS[sign].classes))
        P = g[pc].mean()
        yp = P.to_numpy().argmax(1)
        out = out.assign(gt_class=e.y, gt_class_name=e.y.map(names), pred_class=yp,
                         pred_class_name=pd.Series(yp, index=e.index).map(names), confidence=P.max(axis=1).round(4),
                         correct=e.y.to_numpy() == yp, **{c: P[c].round(4) for c in pc})
    out.sort_values(["exam_id", "eye"]).to_csv(paths.OUTPUT_DIR / f"{sign}_test_pred_vs_gt.csv", index_label="sample_id")


def table_rows(sign, s):
    n = {"sign": sign, "epochs": s["epochs_chosen_on_val"], "test patients": s["test_patients"],
         "test rows": s["test_label_rows"]}
    if SIGNS[sign].kind == "regression":
        ci = lambda k, f: f"{f.format(s[k + '_ci95'][0])}–{f.format(s[k + '_ci95'][1])}"
        return {**n, **{k: s[k] for k in ("mae_mm", "rmse_mm", "within_0_05mm", "pearson_r", "agree_0_2mm")},
                "MAE 95% CI": ci("mae_mm", "{:.3f}"), "within 0.05 95% CI": ci("within_0_05mm", "{:.2f}"),
                "MAE seeds": f"{s['mae_mm_seed_mean']:.3f} ± {s['mae_mm_seed_sd']:.3f}"}
    pct = lambda k: f"{100 * s[k + '_ci95'][0]:.0f}–{100 * s[k + '_ci95'][1]:.0f}"
    return {**n, **{k: round(100 * s[k], 1) for k in ("accuracy", "precision", "recall", "f1")},
            "ACC 95% CI": pct("accuracy"), "F1 95% CI": pct("f1"),
            "F1 seeds": f"{100 * s['f1_seed_mean']:.1f} ± {100 * s['f1_seed_sd']:.1f}"}


def main():
    df = pd.read_csv(paths.LABELS_CSV, low_memory=False)
    done = [s for s in SIGNS if pred_path(s).exists()]
    summ = {s: store(s) for s in done}
    for s in done:
        export(s, df, test_rows(s)[0])
    flat = pd.DataFrame([{k: (f"{v[0]:.4f}-{v[1]:.4f}" if isinstance(v, list) and k.endswith("ci95") else v)
                          for k, v in r.items() if k != "classes"} for r in summ.values()])
    flat.to_csv(paths.OUTPUT_DIR / "metrics" / "all_tests.csv", index=False)
    cls = pd.DataFrame([table_rows(s, summ[s]) for s in done if SIGNS[s].kind == "classification"])
    reg = pd.DataFrame([table_rows(s, summ[s]) for s in done if SIGNS[s].kind == "regression"])
    cls.to_csv(paths.OUTPUT_DIR / "test_metrics_classification.csv", index=False)
    reg.to_csv(paths.OUTPUT_DIR / "test_metrics_tear_meniscus.csv", index=False)
    L = ["# SSL V-JEPA 2.1 ViT-L + LoRA + head — test results", "",
         "One shared patient split (train / val / test); the same test patients for every sign. 5 seeds, epoch chosen "
         "on val only, test scored once at that epoch. Eyes graded twice = two rows. Test numbers = 5-seed ensemble; "
         "precision / recall / F1 = macro over the classes present; `F1 seeds` = per-seed mean ± sd. 95 % CI = "
         "patient-level bootstrap (2,000 resamples).", "",
         "## Classification (test, %)", "", cls.to_markdown(index=False) if len(cls) else "—", "",
         "## Tear meniscus height (test, regression, mm)", "", reg.round(3).to_markdown(index=False) if len(reg) else "—", ""]
    (paths.OUTPUT_DIR / "results.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
