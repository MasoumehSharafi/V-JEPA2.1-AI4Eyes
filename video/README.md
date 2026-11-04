# AI4Eyes Video: V-JEPA2.1 pipeline

V-JEPA2.1-based temporal action detection for three clinical video signs:
**NITBUT**, **Invasive-TBUT**, and **Blink Mechanics**. A frozen V-JEPA2.1
ViT-L/16 encoder (optionally domain-adapted via continued SSL pretraining
on unlabeled AI4Eyes video) feeds a trainable, from-scratch ActionFormer-
style temporal detection head.

The image half of AI4Eyes (corneal staining, hyperemia, tear meniscus,
meibography, lid margin) will land in a sibling `image/` directory later,
sharing this repo's manifest format (see "Manifest schema" below) so a
joint image+video pretraining stage can concatenate both manifests.

## What each test predicts

| Test | Classes | Target | Ground-truth shape |
|---|---|---|---|
| NITBUT | 1: `NITBUT` | One tear-breakup event (start, end) per gradable video; ungradable videos have zero events (kept as background, not dropped) | Single segment or none, per video |
| Invasive-TBUT | 1: `INVASIVE_TBUT` | Same shape as NITBUT, different stimulus/ground truth | Single segment or none, per video |
| Blink Mechanics | 2: `Blink Complete`, `Blink Incomplete` | Every blink in the video, classified as complete or incomplete | Multiple point-in-time events per video, converted to short fixed-width pseudo-segments (see "Label modes" below) — there is no true start/end span in the source annotation |

## Model variant

**`vjepa2_1_vit_large_384`** — V-JEPA 2.1 ViT-L/16, distilled from a
ViT-giant teacher, officially released pretrained at 384px
(`vjepa2_1_vitl_dist_vitG_384.pt`). **The image pipeline should use this
same model/checkpoint family** so a later joint pretraining shares one
encoder lineage.

Despite the 384px release resolution, every stage in this repo — domain-
adaptive continued pretraining and supervised fine-tuning alike — runs at
**224px, 16-frame clips, tubelet_size=2, patch_size=16**. This matches
the actual domain-adaptive-pretraining config this pipeline was built
from (`ai4eyes-allsigns-v192excl-224-16f-50ep...`); the ViT architecture
itself is resolution-agnostic (patch-based position embeddings), so this
is a deliberate compute/throughput choice, not a limitation imposed by
the release checkpoint.

Backbone is **always frozen** during supervised training
(`backbone.freeze: true` in every `configs/supervised/*.yaml`) — only the
head (`src/ai4eyes_video/heads.py`) is trained. There is no LoRA/adapter
fine-tuning of the backbone anywhere in this pipeline.

## ⚠️ Detection head: reimplemented, not vendored

`src/ai4eyes_video/heads.py` is a **from-scratch implementation** of an
anchor-free, point-based temporal action detection head in the style of
ActionFormer (Zhang et al., 2022) — multi-scale 1D feature pyramid,
per-level conv classification/regression towers, center-sampling label
assignment, focal + DIoU loss, greedy NMS. It is **not** a copy of any
specific codebase. This keeps the repo's only external framework
dependency as official `vjepa2` (see "Setup" below) — no second cloned
framework for the detection head. **Consequence: exact metric
reproduction of the numbers in "Results" below is not guaranteed** — the
backbone, data handling, and evaluation logic are faithful to what
actually ran, but the trainable head's exact architecture/hyperparameters
differ from whatever produced those numbers.

## Run order

```
scripts/setup_vjepa.sh                              # once
scripts/prepare_data.sh TEST_TYPE LABEL_MODE         # once per test type
scripts/pretrain_ssl.sh                              # once, shared across all three signs (optional)
scripts/export_encoder.sh CHECKPOINT_TAR OUT_PATH    # only if you ran pretrain_ssl.sh
scripts/train_supervised.sh configs/supervised/<test>.yaml
scripts/evaluate.sh configs/supervised/<test>.yaml CHECKPOINT OUT_DIR
scripts/time_inference.sh configs/supervised/<test>.yaml CHECKPOINT OUT_DIR
```

Or, per test type: `bash scripts/run_all.sh TEST_TYPE LABEL_MODE configs/supervised/<test>.yaml`
(set `RUN_SSL=1` to include the shared SSL stage; it's skipped by default
since one domain-adapted encoder serves all three signs).

Label modes for `prepare_data.sh`: `single-segment` (NITBUT, Invasive-TBUT)
or `point-events` (Blink Mechanics).

All training/eval scripts support `RESUME=1 bash scripts/<script>.sh ...`
to continue from a checkpoint instead of restarting.

## Expected outputs

- `scripts/prepare_data.sh` → a manifest CSV + `<test>.json` annotations file (see schemas below)
- `scripts/pretrain_ssl.sh` → `$AI4EYES_WORK_DIR/ssl_dapt/{latest,best}.pth.tar`
- `scripts/export_encoder.sh` → a small encoder-only `.pt` (loadable by `backbone.VJEPA21Backbone`)
- `scripts/train_supervised.sh` → `$AI4EYES_CHECKPOINT_DIR/<test>/`:
  `last.pt` (resumable: model+optimizer+scheduler+epoch, used by `RESUME=1`),
  `checkpoint-N.pt` (bare model state_dict, saved every `checkpoint_interval` epochs),
  `final.pt` (bare model state_dict from the last epoch — pass **this** to `evaluate.sh`/`time_inference.sh`)
- `scripts/evaluate.sh` → `<out_dir>/test_predictions.json` (persisted predictions) + `<out_dir>/test_metrics.json` (computed from that file, never inline during training)
- `scripts/time_inference.sh` → `<out_dir>/timing_report.json`

## Manifest schema

The unlabelled-data manifest (`src/ai4eyes_video/manifest.py`) is the
join format meant to outlive any one pipeline run. The SSL dataset reads
clips from it today; a later joint image+video pretraining can simply
concatenate a video manifest and an image manifest, since both share this
schema. CSV, one row per media item:

| column | meaning |
|---|---|
| `key` | unique id, e.g. `<exam_id>-<eye>` |
| `media_type` | `video` \| `image` |
| `test_type` | e.g. `NITBUT`, `INVASIVE_TBUT`, `BLINK_MECHANICS` (image test types added later by the image pipeline) |
| `patient_id` | patient identifier |
| `exam_id` | exam identifier |
| `eye` | `OD` \| `OS` \| empty (empty only for media with no laterality) |
| `split` | `train` \| `val` \| `test` — copied verbatim from the shared split CSV, **never recomputed by this package** |
| `relative_path` | path relative to a configured media root — never an absolute path or S3 URI |

The per-test labeled annotation format (`src/ai4eyes_video/annotations.py`)
used by supervised training/eval is a small ActivityNet-style JSON; see
that module's docstring for the exact schema.

**No manifest, split CSV, raw export, or annotation file is committed to
this repo** — the data changes weekly, so these are always read from
paths you supply via the environment variables below.

## Clinical rules (enforced in code, not just convention)

- **OD/OS of one patient are never split across train/val/test.**
  `manifest.validate_od_os_rule` raises (naming the offending patient) if
  this is ever violated — called automatically by `read_manifest` and by
  `prepare_data.py`.
- **The test split is read from a CSV path, never re-drawn.**
  `manifest.read_split_csv` is the only place a split assignment is
  determined; `prepare_data.py` has no train/val/test sampling logic of
  its own.
- **Metrics are computed from persisted prediction files, never inside
  the training loop.** `train.py` never imports `metrics.py`.
  `evaluate.py` is split into an explicit inference phase (writes
  `--predictions-out`) and a scoring phase (reads it back).

## Configuration

No path, bucket, or credential is hardcoded anywhere in this package —
every one of these has a default assuming a local, relative layout, and
every default is overridable by its environment variable.

| Variable | Default | Meaning |
|---|---|---|
| `AI4EYES_DATA_ROOT` | `./data` | Root for local data references |
| `AI4EYES_SPLIT_CSV` | `./data/split.csv` | The shared, authoritative split CSV |
| `AI4EYES_MANIFEST` | `./data/manifest.csv` | Unlabelled media manifest |
| `AI4EYES_ANNOTATIONS_DIR` | `./data/annotations` | Per-test labeled annotation JSON files |
| `AI4EYES_VIDEO_ROOT` | `./data/videos` | Video root that manifest `relative_path`s resolve against |
| `AI4EYES_S3_BUCKET` / `AI4EYES_S3_PREFIX` | empty | Only used by your own download tooling; never hardcoded here |
| `AI4EYES_WORK_DIR` | `./work` | Fast scratch (e.g. an ephemeral NVMe volume) for active runs |
| `AI4EYES_CHECKPOINT_DIR` | `./checkpoints` | Durable checkpoint storage, outside the scratch volume |
| `VJEPA2_REPO` | `./vjepa2` | The cloned `facebookresearch/vjepa2` checkout (`setup_vjepa.sh`) |
| `VJEPA2_CHECKPOINT` | `./checkpoints/vjepa2_1_vitl_dist_vitG_384.pt` | Backbone checkpoint currently selected |
| `RESUME` | `0` | `1` to resume training/pretraining from the last checkpoint |

## Setup

```bash
pip install -r requirements.txt   # installs this package (video/) in editable mode too
bash video/scripts/setup_vjepa.sh # clones facebookresearch/vjepa2 at a pinned commit,
                                   # applies video/patches/, installs vjepa2's OWN
                                   # requirements.txt, downloads the official checkpoint
pip install -r requirements.txt   # re-run: vjepa2's requirements.txt pins nothing
                                   # (e.g. "torch>=2") and can upgrade packages this
                                   # repo pinned a specific version of -- this call
                                   # restores them
```

See `video/patches/README.md` for exactly what's patched on top of the
pinned `vjepa2` commit (`204698b45b3712590f06245fbfba32d3be539812`) and a
known, faithfully-reproduced bug in the upstream-adjacent training script.

## Results

As of **2026-10-28** (the date the checkpoints referenced below were
trained), frozen V-JEPA2.1 ViT-L/16 + ActionFormer-style head, scored on
a 60-patient (120-eye) held-out test split:

| Metric | NITBUT | Invasive-TBUT | Blink Mechanics |
|---|---|---|---|
| Gradability accuracy | 53.3% | 71.7% | — |
| Precision / Recall / F1 | 53.3 / 100 / 69.6% | 71.7 / 100 / 83.5% | — |
| Mean tIoU (true positives) | 0.504 | 0.328 | — |
| Start MAE | 1.48s | 2.40s | — |
| Break-up MAE | 4.44s | 4.42s | — |
| Duration MAE | 3.76s | 2.53s | — |
| Blink Complete: P / R / F1 | — | — | 18.6 / 92.0 / 30.9% |
| Blink Incomplete: P / R / F1 | — | — | 20.4 / 94.2 / 33.5% |
| Blink count exact-match | — | — | 1.7% / 0.0% |

Recall is 100% on both single-class tests because the model never predicts
"ungradable" at the default score threshold (0.1) — every error is a false
alarm, not a missed detection. Blink-mechanics count-matching is weak; this
is an honest number, not rounded up. **These numbers came from the
original (OpenTAD-framework) pipeline, not this repo's reimplemented
head** (see the warning above) — re-run `scripts/evaluate.sh` for current
numbers from this codebase, and update this table from those.

## Known gaps / what to verify before relying on this

- No data-prep script from the original pipeline runs survived — `prepare_data.py` is a fresh, generalized reimplementation of that logic, parameterized by column names since the raw export schema changes over time.
- No launcher script survived for the actual domain-adaptive SSL pretraining run that produced the checkpoint behind the results table — `pretrain_ssl.sh` is newly written, following the same invocation pattern as a sibling pilot-run script that did survive.
- The encoder-extraction step (full SSL checkpoint → encoder-only `.pt`) was done by hand originally — `export_encoder.py` is newly written to do this cleanly.
- `video/patches/README.md` documents one reproduced-faithfully bug in the vendored-via-patch `vjepa2` training script.
