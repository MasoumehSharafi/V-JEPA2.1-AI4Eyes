"""Inference time of the 8 image signs (5-seed ensembles), per eye and per patient.

    python -m ai4eyes_image.timing [--patients 20]   → $AI4EYES_OUTPUT_DIR/timing/{timing_per_case.csv,timing_summary.md}

Models are loaded once (one backbone + 5 LoRA/head weight sets per sign, as in deployment). Per case (one eye, or
one patient = both eyes in one batch):
  read   read + preprocess every sign's image from disk (cornea crop for staining, full frame otherwise), plus
         decoding the lower-lid frame from the lipid-layer video
  infer  8 signs × 5 seeds forward pass + ensemble mean (infer_1seed: seed 0 only, not part of total)
Cases = test patients with every image sign present on both eyes; the first 2 are warm-up and not reported.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from ai4eyes_image import paths
from ai4eyes_image.data.frames import read_frame
from ai4eyes_image.inference import SignModel
from ai4eyes_image.preprocess.banks import make_bank
from ai4eyes_image.signs import SIGNS
from ai4eyes_image.train.supervised import data

WARMUP = 2


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(n_patients):
    device = "cuda"
    df = pd.read_csv(paths.LABELS_CSV, low_memory=False)
    per = {s: data(s, df).query("split == 'test'").drop_duplicates(["exam_id", "eye"]).set_index(["exam_id", "eye"])
           for s in SIGNS}
    common = set.intersection(*(set(p.index) for p in per.values()))
    ref = per["corneal_staining"]
    pts = sorted({ref.loc[k].patient_pid for k in common})
    eyes = {p: sorted(k for k in common if ref.loc[k].patient_pid == p) for p in pts}
    pts = [p for p in pts if len(eyes[p]) == 2][: n_patients + WARMUP]
    frames = pd.read_csv(paths.DATA_DIR / "lower_lid_frames.csv").set_index(["exam_id", "eye"])

    sync(); t0 = time.perf_counter()
    models = {s: SignModel(s, device) for s in SIGNS}
    sync(); t_load = time.perf_counter() - t0
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9

    rows = []
    for i, p in enumerate(pts):
        for unit, keys in [("eye", [[k] for k in eyes[p]]), ("patient", [eyes[p]])]:
            for ks in keys:
                sync(); t0 = time.perf_counter()
                banks, ids = {}, {}
                for s in SIGNS:
                    d = per[s].loc[ks].reset_index()[["exam_id", "eye", "image_path"]]
                    d["sample_id"] = [f"{e}-{y}" for e, y in zip(d.exam_id, d.eye)]
                    banks[s], ids[s] = make_bank(s, d), list(d.sample_id)
                for k in ks:   # lower-lid frame from the video (the bank above reads the saved copy of it)
                    f = frames.loc[k]
                    if Path(f.video_path).exists():
                        read_frame(f.video_path, f.t)
                t1 = time.perf_counter()
                for s, m in models.items():
                    m.predict_bank(banks[s], ids[s])
                sync(); t2 = time.perf_counter()
                for s, m in models.items():
                    m.predict_bank(banks[s], ids[s], n_seeds=1)
                sync(); t3 = time.perf_counter()
                rows.append(dict(patient=p, unit=unit, eyes=len(ks), warmup=i < WARMUP, read_s=t1 - t0,
                                 infer_s=t2 - t1, infer_1seed_s=t3 - t2, total_s=t2 - t0))
    r = pd.DataFrame(rows)
    out = paths.OUTPUT_DIR / "timing"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "timing_per_case.csv", index=False)
    m = r[~r.warmup]
    q = lambda c: m.groupby("unit")[c].agg(median="median", p10=lambda x: x.quantile(.1), p90=lambda x: x.quantile(.9))
    summ = pd.concat({c: q(c) for c in ["read_s", "infer_s", "infer_1seed_s", "total_s"]}, axis=1).round(3)
    summ.columns = [f"{a} {b}" for a, b in summ.columns]
    txt = [f"# Inference time — 8 image signs ({torch.cuda.get_device_name()})", "",
           f"Cases: {m.patient.nunique()} test patients ({len(m[m.unit == 'eye'])} single-eye runs, "
           f"{len(m[m.unit == 'patient'])} two-eye runs), after {WARMUP} warm-up patients. 8 signs × 5 seeds, fp16.", "",
           f"One-off model load: {t_load:.1f} s, peak GPU memory {gpu_gb:.1f} GB.", "", summ.to_markdown(), ""]
    (out / "timing_summary.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=20)
    main(ap.parse_args().patients)
