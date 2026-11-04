#!/usr/bin/env bash
#
# Runs inference on the test split, persists predictions to a JSON file,
# then scores that persisted file against ground truth. Metrics are never
# computed inside the training loop -- this script is the only place that
# calls ai4eyes_video.metrics.
#
# Usage: bash video/scripts/evaluate.sh CONFIG_YAML CHECKPOINT OUT_DIR
#   e.g. bash video/scripts/evaluate.sh video/configs/supervised/nitbut.yaml \
#            checkpoints/nitbut/checkpoint-59.pt outputs/nitbut
set -euo pipefail

CONFIG_YAML="${1:-}"
CHECKPOINT="${2:-}"
OUT_DIR="${3:-}"
if [[ -z "${CONFIG_YAML}" || -z "${CHECKPOINT}" || -z "${OUT_DIR}" ]]; then
    echo "Usage: bash $0 CONFIG_YAML CHECKPOINT OUT_DIR"
    exit 1
fi

mkdir -p "${OUT_DIR}"
python -m ai4eyes_video.evaluate \
    --config "${CONFIG_YAML}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --predictions-out "${OUT_DIR}/test_predictions.json" \
    --metrics-out "${OUT_DIR}/test_metrics.json"
