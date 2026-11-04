#!/usr/bin/env bash
#
# Times model load + per-eye inference latency on up to N test-split videos.
#
# Usage: bash video/scripts/time_inference.sh CONFIG_YAML CHECKPOINT OUT_DIR [MAX_VIDEOS]
set -euo pipefail

CONFIG_YAML="${1:-}"
CHECKPOINT="${2:-}"
OUT_DIR="${3:-}"
MAX_VIDEOS="${4:-20}"
if [[ -z "${CONFIG_YAML}" || -z "${CHECKPOINT}" || -z "${OUT_DIR}" ]]; then
    echo "Usage: bash $0 CONFIG_YAML CHECKPOINT OUT_DIR [MAX_VIDEOS]"
    exit 1
fi

mkdir -p "${OUT_DIR}"
python -m ai4eyes_video.time_inference \
    --config "${CONFIG_YAML}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --max-videos "${MAX_VIDEOS}" \
    --report-out "${OUT_DIR}/timing_report.json"
