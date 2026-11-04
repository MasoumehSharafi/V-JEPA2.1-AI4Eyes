#!/usr/bin/env bash
# V-JEPA 2.1 ViT-L/16 384 (Meta, MIT licence): clone the official code at a pinned commit and keep only the encoder
# and predictor weights of the release checkpoint (~5.2 GB download, ~1.2 GB + ~0.1 GB kept).
#   scripts/setup_vjepa.sh
set -euo pipefail
. "$(dirname "$0")/_env.sh"
COMMIT=204698b45b3712590f06245fbfba32d3be539812      # facebookresearch/vjepa2, 2026-03-23 (V-JEPA 2.1 release)
DIR=${AI4EYES_VJEPA2_DIR:-third_party/vjepa2}
CK=${AI4EYES_CHECKPOINT_DIR:-checkpoints}/vjepa2
if [ ! -d "$DIR/app/vjepa_2_1" ]; then
  git clone -q https://github.com/facebookresearch/vjepa2.git "$DIR"
  git -C "$DIR" checkout -q "$COMMIT"
fi
mkdir -p "$CK"
ENC=$CK/vjepa2_1_vitl_384_encoder.pt
PRED=$CK/vjepa2_1_vitl_384_predictor.pt
FULL=$CK/vjepa2_1_vitl_dist_vitG_384.pt
if [ ! -f "$ENC" ] || [ ! -f "$PRED" ]; then
  [ -f "$FULL" ] || curl -sSf -o "$FULL" https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
  $PY - "$FULL" "$ENC" "$PRED" <<'PYEOF'
import os, sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
clean = lambda sd: {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
for key, out in (("ema_encoder", sys.argv[2]), ("predictor", sys.argv[3])):
    if not os.path.exists(out):
        sd = clean(ck[key])
        torch.save(sd, out)
        print(f"saved {len(sd)} {key} tensors → {out}")
PYEOF
  rm -f "$FULL"
fi
step "V-JEPA 2.1 ready: code $DIR @ ${COMMIT:0:7}, weights $CK"
