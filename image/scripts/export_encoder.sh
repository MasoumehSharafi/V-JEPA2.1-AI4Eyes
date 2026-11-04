#!/usr/bin/env bash
# EMA target encoder of the finished SSL run → the backbone file the supervised config points to.
#   scripts/export_encoder.sh
set -euo pipefail
. "$(dirname "$0")/_env.sh"
CFG=${CFG:-configs/pretrain_ssl.yaml}
NAME=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFG")
CK=${AI4EYES_CHECKPOINT_DIR:-checkpoints}
$PY -m ai4eyes_image.pretrain.vjepa_ssl export "$CK/ssl/${NAME}_last.pt" "$CK/vjepa2/${NAME}_encoder.pt"
