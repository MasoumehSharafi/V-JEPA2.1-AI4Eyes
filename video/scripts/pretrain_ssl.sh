#!/usr/bin/env bash
#
# Domain-adaptive continued SSL pretraining of V-JEPA2.1 ViT-L/16 on the
# combined unlabeled video pool (manifest rows NOT in the `test` split).
# Converts our manifest to vjepa2's own CSV format, substitutes paths into
# configs/ssl/pretrain_dapt.yaml, then runs vjepa2's own trainer
# (app.main) -- we do not reimplement V-JEPA's SSL training loop.
#
# Usage: bash video/scripts/pretrain_ssl.sh
# RESUME=1 bash video/scripts/pretrain_ssl.sh   # resume via vjepa2's own
#                                                 # load_checkpoint/latest.pth.tar
#
# Env vars (see video/README.md):
#   AI4EYES_MANIFEST       manifest CSV to pretrain on (required)
#   AI4EYES_VIDEO_ROOT     video root the manifest's relative_path is under
#   VJEPA2_REPO            cloned vjepa2 checkout (default ./vjepa2)
#   VJEPA2_CHECKPOINT      official release checkpoint to initialize from
#   AI4EYES_WORK_DIR       output folder for the SSL run (fast/ephemeral scratch ok)
set -euo pipefail

AI4EYES_MANIFEST="${AI4EYES_MANIFEST:?Set AI4EYES_MANIFEST to the manifest CSV to pretrain on}"
AI4EYES_VIDEO_ROOT="${AI4EYES_VIDEO_ROOT:-./data/videos}"
VJEPA2_REPO="${VJEPA2_REPO:-./vjepa2}"
VJEPA2_CHECKPOINT="${VJEPA2_CHECKPOINT:-./checkpoints/vjepa2_1_vitl_dist_vitG_384.pt}"
AI4EYES_WORK_DIR="${AI4EYES_WORK_DIR:-./work}/ssl_dapt"
RESUME="${RESUME:-0}"

mkdir -p "${AI4EYES_WORK_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Converting manifest to vjepa2's training-list format =="
VJEPA_TRAIN_CSV="${AI4EYES_WORK_DIR}/vjepa_train.csv"
python -m ai4eyes_video.export_vjepa_manifest \
    --manifest "${AI4EYES_MANIFEST}" --video-root "${AI4EYES_VIDEO_ROOT}" \
    --out "${VJEPA_TRAIN_CSV}"

echo "== Rendering configs/ssl/pretrain_dapt.yaml with real paths =="
RENDERED_CONFIG="${AI4EYES_WORK_DIR}/pretrain_dapt.yaml"
sed \
    -e "s#__OUTPUT_FOLDER__#${AI4EYES_WORK_DIR}#g" \
    -e "s#__DATASET_CSV__#${VJEPA_TRAIN_CSV}#g" \
    -e "s#__PRETRAINED_ENCODER__#${VJEPA2_CHECKPOINT}#g" \
    "${SCRIPT_DIR}/../configs/ssl/pretrain_dapt.yaml" > "${RENDERED_CONFIG}"

if [[ "${RESUME}" == "1" ]]; then
    echo "RESUME=1: vjepa2's trainer auto-resumes from ${AI4EYES_WORK_DIR}/latest.pth.tar if present (meta.load_checkpoint: true)."
fi

echo "== Launching vjepa2's own trainer (app.main) =="
cd "${VJEPA2_REPO}"
python -m app.main --fname "${RENDERED_CONFIG}" --devices cuda:0

echo
echo "Done. Checkpoints under ${AI4EYES_WORK_DIR}:"
echo "  latest.pth.tar  -- most recent epoch"
echo "  best.pth.tar    -- see video/patches/README.md for a caveat on exactly when this is written"
echo "Next: scripts/export_encoder.sh to extract just the encoder for supervised training."
