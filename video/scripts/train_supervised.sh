#!/usr/bin/env bash
#
# Trains the ActionFormer-style head on top of a frozen V-JEPA2.1 backbone
# for one test type.
#
# Usage: bash video/scripts/train_supervised.sh CONFIG_YAML
#   e.g. bash video/scripts/train_supervised.sh video/configs/supervised/nitbut.yaml
#
# RESUME=1 bash video/scripts/train_supervised.sh CONFIG_YAML   # resume from last.pt
set -euo pipefail

CONFIG_YAML="${1:-}"
if [[ -z "${CONFIG_YAML}" ]]; then
    echo "Usage: bash $0 CONFIG_YAML"
    exit 1
fi

RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG=(--resume)
fi

python -m ai4eyes_video.train --config "${CONFIG_YAML}" "${RESUME_FLAG[@]}"
