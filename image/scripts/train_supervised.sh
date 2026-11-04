#!/usr/bin/env bash
# LoRA + head on the SSL encoder, every image sign (or --signs ...): 5 seeds × 30 epochs, epoch chosen on val.
# ~9-10 h for all 8 signs on 1 A10G. Resumable: finished signs are skipped.
#   scripts/train_supervised.sh [--signs corneal_staining tear_meniscus] [--max-epochs 2 --seeds 0]
set -euo pipefail
. "$(dirname "$0")/_env.sh"
step "supervised training (configs/supervised.yaml)"
$PY -u -m ai4eyes_image.train.supervised run configs/supervised.yaml "$@"
