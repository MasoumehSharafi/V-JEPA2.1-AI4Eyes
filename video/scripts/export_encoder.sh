#!/usr/bin/env bash
#
# Extracts the EMA encoder from a full SSL training checkpoint into a small
# file backbone.VJEPA21Backbone can load directly for supervised training.
#
# Usage: bash video/scripts/export_encoder.sh CHECKPOINT_TAR OUT_PATH
#   e.g. bash video/scripts/export_encoder.sh ./work/ssl_dapt/best.pth.tar \
#            ./checkpoints/vjepa_dapt_encoder.pt
set -euo pipefail

CHECKPOINT_TAR="${1:-}"
OUT_PATH="${2:-}"
if [[ -z "${CHECKPOINT_TAR}" || -z "${OUT_PATH}" ]]; then
    echo "Usage: bash $0 CHECKPOINT_TAR OUT_PATH"
    exit 1
fi

python -m ai4eyes_video.export_encoder --checkpoint "${CHECKPOINT_TAR}" --key ema_encoder --out "${OUT_PATH}"
