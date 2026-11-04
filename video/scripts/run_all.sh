#!/usr/bin/env bash
#
# Chains the full pipeline for one test type: setup -> prepare_data ->
# [pretrain_ssl, once, shared across all three signs] -> export_encoder
# -> train_supervised -> evaluate -> time_inference.
#
# Usage: bash video/scripts/run_all.sh TEST_TYPE LABEL_MODE CONFIG_YAML
#   e.g. bash video/scripts/run_all.sh NITBUT single-segment video/configs/supervised/nitbut.yaml
#
# RUN_SSL=1   also (re-)run the shared SSL pretraining stage first (skip by
#             default since one domain-adapted encoder covers all three
#             signs -- running it three times would be wasteful).
# RESUME=1    forwarded to train_supervised.sh and pretrain_ssl.sh.
set -euo pipefail

TEST_TYPE="${1:-}"
LABEL_MODE="${2:-}"
CONFIG_YAML="${3:-}"
if [[ -z "${TEST_TYPE}" || -z "${LABEL_MODE}" || -z "${CONFIG_YAML}" ]]; then
    echo "Usage: bash $0 TEST_TYPE LABEL_MODE CONFIG_YAML"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "### 1/6 setup_vjepa.sh"
bash "${SCRIPT_DIR}/setup_vjepa.sh"

echo "### 2/6 prepare_data.sh"
bash "${SCRIPT_DIR}/prepare_data.sh" "${TEST_TYPE}" "${LABEL_MODE}"

if [[ "${RUN_SSL:-0}" == "1" ]]; then
    echo "### 3/6 pretrain_ssl.sh (RUN_SSL=1)"
    bash "${SCRIPT_DIR}/pretrain_ssl.sh"
    echo "### 3b/6 export_encoder.sh"
    bash "${SCRIPT_DIR}/export_encoder.sh" \
        "${AI4EYES_WORK_DIR:-./work}/ssl_dapt/best.pth.tar" \
        "${AI4EYES_CHECKPOINT_DIR:-./checkpoints}/vjepa_dapt_encoder.pt"
else
    echo "### 3/6 pretrain_ssl.sh skipped (set RUN_SSL=1 to include it -- it's shared across all three signs)"
fi

CHECKPOINT_DIR="${AI4EYES_CHECKPOINT_DIR:-./checkpoints}"
OUT_DIR="./outputs/$(echo "${TEST_TYPE}" | tr '[:upper:]' '[:lower:]')"

echo "### 4/6 train_supervised.sh"
bash "${SCRIPT_DIR}/train_supervised.sh" "${CONFIG_YAML}"

FINAL_CHECKPOINT="${CHECKPOINT_DIR}/$(echo "${TEST_TYPE}" | tr '[:upper:]' '[:lower:]')/final.pt"

echo "### 5/6 evaluate.sh"
bash "${SCRIPT_DIR}/evaluate.sh" "${CONFIG_YAML}" "${FINAL_CHECKPOINT}" "${OUT_DIR}"

echo "### 6/6 time_inference.sh"
bash "${SCRIPT_DIR}/time_inference.sh" "${CONFIG_YAML}" "${FINAL_CHECKPOINT}" "${OUT_DIR}"

echo
echo "Done. See ${OUT_DIR}/ for predictions, metrics, and the timing report."
