#!/usr/bin/env bash
# Continued V-JEPA 2.1 SSL pretraining on the pooled unlabelled images (configs/pretrain_ssl.yaml, ~2.7 h on 1 A10G).
# The checkpoint is written to $AI4EYES_WORK_DIR/ssl (fast disk) and copied to $AI4EYES_CHECKPOINT_DIR/ssl every
# `mirror_every` epochs, so a wiped scratch disk loses at most that many epochs.
#   scripts/pretrain_ssl.sh            # start
#   RESUME=1 scripts/pretrain_ssl.sh   # continue (falls back to the persistent copy if the scratch copy is gone)
set -euo pipefail
. "$(dirname "$0")/_env.sh"
CFG=${CFG:-configs/pretrain_ssl.yaml}
NAME=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFG")
WORK=${AI4EYES_WORK_DIR:-work}/ssl
KEEP=${AI4EYES_CHECKPOINT_DIR:-checkpoints}/ssl
mkdir -p "$WORK" "$KEEP"
if [ "${RESUME:-0}" = 1 ] && [ ! -f "$WORK/${NAME}_last.pt" ] && [ -f "$KEEP/${NAME}_last.pt" ]; then
  step "scratch checkpoint missing; resuming from $KEEP"
  cp "$KEEP/${NAME}_last.pt" "$KEEP/${NAME}_log.csv" "$WORK/"
fi
step "V-JEPA 2.1 SSL ($CFG) → $WORK, copies → $KEEP"
$PY -u -m ai4eyes_image.pretrain.vjepa_ssl "$CFG" --out-dir "$WORK" --mirror-dir "$KEEP" \
    $([ "${RESUME:-0}" = 1 ] && echo --resume)
step "done → $KEEP/${NAME}_last.pt (next: scripts/export_encoder.sh)"
