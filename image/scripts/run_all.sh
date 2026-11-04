#!/usr/bin/env bash
# The whole image pipeline, in order. Each stage skips work that is already done, so a re-run resumes.
#   scripts/run_all.sh                 (RESUME=1 to continue an interrupted SSL run)
set -euo pipefail
cd "$(dirname "$0")"
./setup_vjepa.sh
./prepare_data.sh
./build_ssl_manifest.sh
. ./_env.sh
ENC=${AI4EYES_CHECKPOINT_DIR:-checkpoints}/vjepa2/vjepa_dapt_all_signs_v192_encoder.pt
if [ ! -f "$ENC" ]; then
  scripts/pretrain_ssl.sh
  scripts/export_encoder.sh
fi
scripts/train_supervised.sh
scripts/evaluate.sh
