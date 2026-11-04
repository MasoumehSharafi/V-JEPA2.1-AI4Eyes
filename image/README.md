# Image signs: V-JEPA 2.1 SSL + LoRA

This folder holds the code for the 8 photo-based signs of the dry-eye exam. It starts from the released
**V-JEPA 2.1 ViT-L/16** (Meta) and does three things:

1. Continues self-supervised pretraining on the unlabelled photos of all signs, pooled.
2. Trains a **LoRA + linear head** per sign on the labelled exams.
3. Scores each sign once on a held-out test set.

The video signs (NITBUT, invasive TBUT, blink) live in [`../video`](../video). Both folders use the same patient split, so
the test patients are the same for every sign.

| Sign | Photo | Target |
|---|---|---|
| `corneal_staining` | fluorescein staining (cornea crop, fixation ring inpainted) | 0 / 1 / ≥2 or ungradable |
| `hyperemia_nasal` | meniscus photo | 1 / 2 / ≥3 |
| `hyperemia_temporal` | meniscus photo | ≤1 / 2 / 3 |
| `meibography_lower` | infrared meibography, lower lid | meiboscale 0 / 1 / 2 / 3+4+ungradable |
| `meibography_upper` | infrared meibography, upper lid | 0+1 / 2 / 3+ungradable |
| `upper_lid_margin` | lid-margin photo | none / blepharitis / vascular / other |
| `lower_lid_margin` | one frame of the lipid-layer video | none / MGD-blepharitis / vascular / other |
| `tear_meniscus` | meniscus photo | tear meniscus height in mm (regression) |

The class rules (merges, ungradable handling, lid-margin condition groups) are in
[`ai4eyes_image/signs.py`](ai4eyes_image/signs.py).

## Results (test set: 60 patients)

Each number is from a 5-seed ensemble. The epoch was chosen on the 30 validation patients only, and the test set was scored once. Precision, recall and F1
are macro averages over the classes present. The 95 % CIs come from a patient-level bootstrap.

| Sign | ACC % | Precision % | Recall % | F1 % | F1 95 % CI |
|---|---:|---:|---:|---:|---|
| Corneal staining | 58.6 | 50.5 | 49.3 | 49.4 | 41–57 |
| Hyperemia nasal | 62.9 | 53.2 | 50.9 | 50.5 | 41–58 |
| Hyperemia temporal | 64.2 | 48.9 | 48.8 | 48.8 | 40–58 |
| Meibography lower lid | 42.8 | 36.6 | 36.5 | 35.3 | 26–45 |
| Meibography upper lid | 62.0 | 59.8 | 58.9 | 59.3 | 50–67 |
| Upper lid margin | 54.6 | 43.0 | 40.2 | 40.4 | 30–50 |
| Lower lid margin | 40.4 | 36.3 | 36.6 | 36.4 | 25–47 |

On the same 60 test patients, tear meniscus height reached MAE 0.083 mm (95 % CI 0.066–0.103), RMSE 0.126 mm and
Pearson r 0.41. 39 % of eyes were within ±0.05 mm, and 86 % agreed with the truth at the 0.2 mm dry-eye cut-off.

Inference on one A10G takes about 5.7 s per eye for all 8 signs × 5 seeds, of which 4.4 s is the forward passes. Loading the models takes 41 s once and uses 9.9 GB of GPU memory.

## Pipeline

```
labels CSV ──► prepare_data.sh ──► media in data/images, lower-lid frames, cornea/ring masks, shared split
S3 (unlabelled) ──► build_ssl_manifest.sh ──► ssl_manifest.csv (every unlabelled image, labelled exams excluded)
released V-JEPA 2.1 ViT-L ──► pretrain_ssl.sh ──► export_encoder.sh ──► SSL encoder
SSL encoder + labels ──► train_supervised.sh ──► per-sign LoRA + head (5 seeds), val/test predictions
predictions ──► evaluate.sh ──► results.md, metrics/<sign>/*, per-eye CSVs
```

| Step | Script | Time (1 × A10G) |
|---|---|---|
| 1. Code + released weights | `scripts/setup_vjepa.sh` | ~5 min (5.2 GB download) |
| 2. Labelled data + split | `scripts/prepare_data.sh` | ~10 min |
| 3. SSL manifest | `scripts/build_ssl_manifest.sh` | ~10 min |
| 4. SSL pretraining, 50 epochs | `scripts/pretrain_ssl.sh` (`RESUME=1` to continue) | ~2.7 h |
| 5. Export the encoder | `scripts/export_encoder.sh` | 1 min |
| 6. Supervised, 8 signs × 5 seeds × 30 epochs | `scripts/train_supervised.sh` | ~9–10 h |
| 7. Test metrics | `scripts/evaluate.sh` | 2 min |
| Inference timing | `scripts/time_inference.sh` | 3 min |

`scripts/run_all.sh` runs steps 1–7 in order. Every step skips work that is already done, so a re-run picks up where it stopped.

### What each step does

- **SSL pretraining** ([`pretrain/vjepa_ssl.py`](ai4eyes_image/pretrain/vjepa_ssl.py)) uses the V-JEPA 2.1 image objective in
  self-distillation form:
  - The context encoder sees the visible tokens of a 10-block mask.
  - The predictor regresses the EMA target encoder's tokens at the masked positions (L1), plus the dense context loss.
  - The released predictor body is kept. Its two output heads are replaced, because the released ViT-L was distilled from ViT-G.

  Settings: 384 px, batch 16, 50 epochs, encoder learning rate 2e-5. The EMA target encoder becomes the backbone.
- **Supervised training** ([`train/supervised.py`](ai4eyes_image/train/supervised.py)) trains LoRA on attention q and v of every block, plus a linear head.
  - The head starts random and trains jointly with LoRA. The backbone weights stay frozen.
  - Input is 448 px, with masked mean pooling over the tokens inside the crop.
  - Classification uses class-weighted cross-entropy. Tear meniscus uses MSE on the standardised height.
  - One recipe for every sign ([`configs/supervised.yaml`](configs/supervised.yaml)). Each sign trains with 5 seeds for 30 epochs; the epoch is picked on val (macro-F1, or MAE for tear meniscus) and the 5 checkpoints of that epoch are kept.
- **Evaluation** ([`eval/report.py`](ai4eyes_image/eval/report.py)) computes every metric from the persisted prediction files,
  never inside the training loop.

## Setup

```bash
cd image
python -m venv .venv && . .venv/bin/activate          # Python 3.10
pip install -r requirements.txt
cp env.example .env                                    # then edit paths and AI4EYES_S3_BUCKET
scripts/setup_vjepa.sh
pytest tests                                           # no data or GPU needed
```

Every script sources `image/.env`. All locations come from environment variables, and their defaults are git-ignored
folders inside `image/`:

| Variable | Default | What |
|---|---|---|
| `AI4EYES_LABELS_CSV` | `data/labels.csv` | labels, one row per exam × eye × test type × labeller |
| `AI4EYES_SPLIT_CSV` | `data/split.csv` | `patient_pid,split`: the file shared with the video repo |
| `AI4EYES_IMAGES_DIR` | `data/images` | media, `<exam_id>/<file>` as in S3 |
| `AI4EYES_SSL_MANIFEST` | `data/ssl_manifest.csv` | unlabelled images for SSL |
| `AI4EYES_CHECKPOINT_DIR` | `checkpoints` | `vjepa2/` released + SSL encoders, `ssl/` SSL run, `signs/<sign>/` LoRA + head |
| `AI4EYES_OUTPUT_DIR` | `outputs` | predictions, reports, metrics, timing |
| `AI4EYES_WORK_DIR` | `work` | scratch for per-epoch weights and the live SSL checkpoint (use a fast local disk) |
| `AI4EYES_VJEPA2_DIR` | `third_party/vjepa2` | official V-JEPA 2 code (cloned by `setup_vjepa.sh`) |
| `AI4EYES_S3_BUCKET` | (none) | bucket with the exam media; read access is enough |

### Labels CSV

The labels CSV needs these columns:

- **For every row:** `sample_id`, `patient_pid`, `exam_id`, `eye` (OD/OS), `test_type`, `eye_status` (COMPLETE / REJECTED),
  `media_s3_key`, `labeller_pid`.
- **Label columns:** `corneal_staining_van_bijsterveld` (0–5 reference-panel grade), `hyperemia_nasal_grade`,
  `hyperemia_temporal_grade`, `meibography_meiboscale`, `upper_lid_margin_conditions`, `lower_lid_margin_conditions`
  (JSON lists), `meniscus_height_mm`.
- **For lower lid margin:** `annotations` (JSON with the labeller's "Selected" video frame).

An eye graded by two labellers has two rows, and training and testing both count it twice.

### Split

`ai4eyes_image.data.split` draws 60 test, 30 val and the rest train *patients*, balancing every sign's class
mix. Both eyes and every exam of a patient always land in the same split. The draw is deterministic. If
`AI4EYES_SPLIT_CSV` exists, the step only checks that it still matches. It also writes `<split>_eyes.csv`, with every exam / eye, its split and its media keys, for the video repo.

## SSL manifest

The manifest has one row per unlabelled image. The same schema is meant for a later **joint image + video pretraining**,
which can concatenate manifests:

| Column | Meaning |
|---|---|
| `key` | S3 key, or a relative id for local files |
| `exam_id`, `eye` | as in the labels CSV; any exam that appears there is excluded |
| `split` | train / val, from an md5 hash of `exam_id` (V-JEPA SSL trains on all rows) |
| `ring_ok`, `ring_cx`, `ring_cy`, `ring_r`, `ring_dots` | fixation-ring fit (staining only; inpainted when loaded) |
| `crop_ok`, `box_x0`, `box_y0`, `box_side` | square crop box in native pixels (may extend past the frame and be zero-padded) |
| `image_type` | `staining`, `meniscus`, `meibography`, `lid_margin` or `lower_lid` |
| `local_path` | set when the image is read from disk instead of S3 (lower-lid frames from the lipid-layer videos) |

The published run used 2,608 images:

| Image type | Images |
|---|---:|
| staining | 426 |
| meniscus | 442 |
| meibography | 843 |
| lid margin | 438 |
| lower-lid frames | 459 |

S3 keeps growing, so a rebuild picks up newer uploads as well.

## Reproducibility

This code was extracted from the research code that produced the results above, and checked against it on the
same data:

- **Split, labels and crops:** the shared split, every sign's labels, the image crops and the augmented training views are identical.
- **SSL:** the inputs are identical, and the first 3 losses match to 1e-6.
- **Inference:** loading the published checkpoints gives bit-identical outputs to the original code.
- **Training:** one training epoch matches the original run within 5e-4 (fp16 GPU non-determinism).
- **Evaluation:** the evaluation reproduces every published test metric exactly, including the bootstrap CIs.

Two details matter for exact reproduction:

- **Lower-lid frames:** an eye graded twice can have two "Selected" frame times. The published models took the first
  labelled release's rows first: `ai4eyes_image.data.frames --priority-ids <file of those sample_ids>`.
- **Cornea mask:** a classical ray-cast boundary, not a trained segmenter
  ([`preprocess/masks.py`](ai4eyes_image/preprocess/masks.py)).

## Layout

```
image/
  ai4eyes_image/
    paths.py, config.py, signs.py     locations, config merge, the 8 signs and their label rules
    data/       s3.py, download.py, frames.py, split.py
    preprocess/ masks.py (ring + cornea), transforms.py (crop, augmentation, token mask), banks.py
    models/     vjepa.py (encoder on still images), lora.py (LoRA + head)
    pretrain/   manifest.py, dataset.py, vjepa_ssl.py (SSL + encoder export)
    train/      trainer.py (seeded setup, epoch, predict), supervised.py
    eval/       metrics.py, report.py
    inference.py, timing.py
  configs/      pretrain_ssl.yaml, supervised.yaml
  scripts/      one bash script per step (+ run_all.sh)
  tests/        unit tests on synthetic data
```

## Licences

- **V-JEPA 2.1 code and weights:** MIT (facebookresearch/vjepa2, cloned at a pinned commit by `setup_vjepa.sh` and not
  vendored here).
- **This code:** depends only on permissively licensed packages, so it can be used commercially.
