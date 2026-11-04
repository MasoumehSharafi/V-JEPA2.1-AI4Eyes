#!/usr/bin/env bash
# Inference time of the 8 image signs per eye / per patient (needs the trained checkpoints and a GPU).
#   scripts/time_inference.sh [--patients 20]
set -euo pipefail
. "$(dirname "$0")/_env.sh"
$PY -m ai4eyes_image.timing "$@"
