#!/usr/bin/env bash
#
# Builds the unlabelled manifest + per-test annotations.json for one test
# type from the raw clinical export + the shared split CSV. The split is
# read, never redrawn (see ai4eyes_video/prepare_data.py).
#
# Usage: bash video/scripts/prepare_data.sh TEST_TYPE LABEL_MODE
#   TEST_TYPE   NITBUT | INVASIVE_TBUT | BLINK_MECHANICS
#   LABEL_MODE  single-segment | point-events
#
# Env vars (see video/README.md for the full table):
#   AI4EYES_RAW_CSV      path to the raw clinical export (required)
#   AI4EYES_SPLIT_CSV    path to the shared split CSV (required)
#   AI4EYES_MANIFEST     output manifest path (default ./data/manifest_<test>.csv)
#   AI4EYES_ANNOTATIONS_DIR  output annotations dir (default ./data/annotations)
set -euo pipefail

TEST_TYPE="${1:-}"
LABEL_MODE="${2:-}"
if [[ -z "${TEST_TYPE}" || -z "${LABEL_MODE}" ]]; then
    echo "Usage: bash $0 TEST_TYPE LABEL_MODE   (LABEL_MODE in single-segment, point-events)"
    exit 1
fi

AI4EYES_RAW_CSV="${AI4EYES_RAW_CSV:?Set AI4EYES_RAW_CSV to the raw clinical export CSV path}"
AI4EYES_SPLIT_CSV="${AI4EYES_SPLIT_CSV:?Set AI4EYES_SPLIT_CSV to the shared split CSV path}"
AI4EYES_MANIFEST="${AI4EYES_MANIFEST:-./data/manifest_$(echo "${TEST_TYPE}" | tr '[:upper:]' '[:lower:]').csv}"
AI4EYES_ANNOTATIONS_DIR="${AI4EYES_ANNOTATIONS_DIR:-./data/annotations}"
OUT_ANNOTATIONS="${AI4EYES_ANNOTATIONS_DIR}/$(echo "${TEST_TYPE}" | tr '[:upper:]' '[:lower:]').json"

# AI4EYES_CLASSES: space-separated class names, required for point-events
# mode (e.g. "Blink Complete" "Blink Incomplete"), e.g.:
#   AI4EYES_CLASSES='Blink Complete,Blink Incomplete'
EXTRA_ARGS=()
if [[ -n "${AI4EYES_CLASSES:-}" ]]; then
    IFS=',' read -r -a CLASS_ARRAY <<< "${AI4EYES_CLASSES}"
    EXTRA_ARGS=(--classes "${CLASS_ARRAY[@]}")
fi

python -m ai4eyes_video.prepare_data \
    --raw-csv "${AI4EYES_RAW_CSV}" \
    --split-csv "${AI4EYES_SPLIT_CSV}" \
    --test-type "${TEST_TYPE}" \
    --label-mode "${LABEL_MODE}" \
    --out-manifest "${AI4EYES_MANIFEST}" \
    --out-annotations "${OUT_ANNOTATIONS}" \
    "${EXTRA_ARGS[@]}"
