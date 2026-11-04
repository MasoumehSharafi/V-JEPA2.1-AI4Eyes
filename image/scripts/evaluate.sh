#!/usr/bin/env bash
# Test metrics of every trained sign from the persisted predictions → $AI4EYES_OUTPUT_DIR/{results.md,metrics/}.
#   scripts/evaluate.sh
set -euo pipefail
. "$(dirname "$0")/_env.sh"
step "test report + per-test metric files"
$PY -m ai4eyes_image.eval.report
